// src/cache_tier.rs

/// Represents the symbolic cache tier an expert is associated with.
///
/// These tiers help the `MoEOrchestrator` make decisions about expert activation
/// based on system resources and policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    /// Level 1 cache tier, typically for the fastest, smallest experts.
    L1,
    /// Level 2 cache tier.
    L2,
    /// Level 3 cache tier.
    L3,
    /// Represents experts that might be considered RAM-heavy or don't fit typical CPU cache tiers.
    RAM,
}

/// A trait for experts that are tagged with a specific `CacheTier`.
///
/// This allows the system to identify the expected cache/resource footprint of an expert.
pub trait ExpertTagged {
    /// Returns the symbolic `CacheTier` associated with this expert.
    fn cache_tier(&self) -> CacheTier;
}

// Import Expert trait for type constraint in filter_experts_by_tier
use crate::moe::Expert;

/// Filters a list of scored experts based on allowed cache tiers.
///
/// Given a list of expert indices and their scores, a complete list of expert objects,
/// and a list of desired cache tiers, this function returns a new list containing
/// only those experts (index and score) whose cache tier is among the allowed ones.
///
/// # Arguments
///
/// * `indexed_scores`: A slice of tuples, where each tuple contains the original
///   index of an expert and its gating score.
/// * `all_experts`: A slice of boxed `Expert` trait objects. Each expert must also
///   implement `ExpertTagged`.
/// * `allowed_tiers`: A slice of `CacheTier` enums representing the tiers
///   that are currently permitted.
///
/// # Returns
///
/// A `Vec<(usize, f32)>` containing the `(original_index, score)` for experts
/// that match the allowed cache tiers. If `allowed_tiers` is empty, this
/// will always return an empty vector.
pub fn filter_experts_by_tier(
    indexed_scores: &[(usize, f32)],
    all_experts: &[Box<dyn Expert>],
    allowed_tiers: &[CacheTier],
) -> Vec<(usize, f32)> {
    if allowed_tiers.is_empty() {
        // If no tiers are allowed, no experts can be selected.
        return Vec::new();
    }

    let mut filtered_experts = Vec::new();

    for &(expert_original_index, score) in indexed_scores {
        if expert_original_index < all_experts.len() {
            let expert = &all_experts[expert_original_index];
            if allowed_tiers.contains(&expert.cache_tier()) {
                filtered_experts.push((expert_original_index, score));
            }
        } else {
            // This case should ideally not happen if indexed_scores is valid.
            // Log an error or handle as appropriate if it can occur.
            eprintln!(
                "Warning: Expert index {} out of bounds (total experts: {}). Skipping.",
                expert_original_index,
                all_experts.len()
            );
        }
    }
    filtered_experts
}

#[cfg(test)]
mod tests {
    use super::*;
    // Assuming MLPExpert and SymbolicExpert are accessible for testing via crate::moe
    // and implement Expert + ExpertTagged.
    // Their `new` methods also need to be public.
    use crate::moe::{Expert, MLPExpert, SymbolicExpert};


    #[test]
    fn test_filter_single_allowed_tier_l1() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)), // L1
            Box::new(MLPExpert::new("MlpL2")),          // L2
        ];
        let indexed_scores = vec![(0, 0.6), (1, 0.4)];
        let allowed_tiers = [CacheTier::L1];

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], (0, 0.6)); // Only SymbolicExpert (index 0)
    }

    #[test]
    fn test_filter_multiple_allowed_tiers_l1_l2() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)), // L1 (idx 0)
            Box::new(MLPExpert::new("MlpL2")),          // L2 (idx 1)
        ];
        let indexed_scores = vec![(0, 0.6), (1, 0.4)];
        let allowed_tiers = [CacheTier::L1, CacheTier::L2];

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&(0, 0.6)));
        assert!(filtered.contains(&(1, 0.4)));
    }

    #[test]
    fn test_filter_no_experts_match() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)), // L1
            Box::new(MLPExpert::new("MlpL2")),          // L2
        ];
        let indexed_scores = vec![(0, 0.6), (1, 0.4)];
        let allowed_tiers = [CacheTier::L3]; // No L3 experts defined

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_allowed_tiers_is_empty() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)),
            Box::new(MLPExpert::new("MlpL2")),
        ];
        let indexed_scores = vec![(0, 0.6), (1, 0.4)];
        let allowed_tiers = []; // Empty allowed tiers

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_indexed_scores_is_empty() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)),
            Box::new(MLPExpert::new("MlpL2")),
        ];
        let indexed_scores = []; // Empty scores
        let allowed_tiers = [CacheTier::L1, CacheTier::L2];

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        assert_eq!(filtered.len(), 0);
    }
    
    #[test]
    fn test_filter_out_of_bounds_index_ignored() {
        let experts: Vec<Box<dyn Expert>> = vec![
            Box::new(SymbolicExpert::new("SymL1", 0.5)), // Index 0, L1
            Box::new(MLPExpert::new("MlpL2")),          // Index 1, L2
        ];
        let indexed_scores = vec![(0, 0.5), (1, 0.3), (2, 0.2)]; 
        let allowed_tiers = [CacheTier::L1, CacheTier::L2];

        let filtered = filter_experts_by_tier(&indexed_scores, &experts, &allowed_tiers);
        
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&(0, 0.5))); 
        assert!(filtered.contains(&(1, 0.3))); 
    }
}
