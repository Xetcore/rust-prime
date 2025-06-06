// Renaming import for clarity, though not strictly necessary if TruncationParams and PaddingParams are not used elsewhere.
// For this file, direct use of tokenizers::TruncationParams and tokenizers::PaddingParams in method signatures is fine.
use tokenizers::{
    Tokenizer, 
    Error as TokenizersLibraryError, 
    PaddingParams, 
    TruncationParams, 
    AddedToken,
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPretokenizer,
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    // normalizers::utils::Sequence as NormalizerSequence, // For potential NFC normalizer
    // normalizers::unicode::NFC,
};
use std::path::{Path, PathBuf}; // Added PathBuf for error context
use std::fmt;
use std::io; // For std::io::Error

#[cfg(feature = "tokenizer-debug-logs")]
use log::{debug, trace, warn};

/// Represents errors that can occur within the `TokenizerWrapper`.
#[derive(Debug)]
pub enum TokenizerError {
    /// An I/O error occurred.
    Io(io::Error),
    /// An error originating from the underlying `tokenizers` library.
    /// Contains the string representation of the library error.
    Library(String),
    /// Error encountered while attempting to load a tokenizer model from a file.
    /// This variant can be used if specific context beyond the library error is needed.
    /// For direct `?` usage with `tokenizers::Error`, `Library` variant will be used.
    FailedToLoad { path: PathBuf, source_message: String },
    /// Error encountered during the text encoding process.
    EncodingFailed { text: String, source_message: String },
    /// Error encountered during the token ID decoding process.
    DecodingFailed { ids: Vec<u32>, source_message: String },
}

/// Options for configuring the encoding process, specifically truncation and padding.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// Optional truncation parameters. If `None`, no truncation is applied beyond tokenizer defaults.
    pub truncation: Option<TruncationParams>,
    /// Optional padding parameters. If `None`, no padding is applied.
    pub padding: Option<PaddingParams>,
}

impl EncodeOptions {
    /// Creates new `EncodeOptions` with both truncation and padding set to `None`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(err) => write!(f, "I/O error: {}", err),
            TokenizerError::Library(msg) => write!(f, "Tokenizer library error: {}", msg),
            TokenizerError::FailedToLoad{ path, source_message } => write!(f, "Failed to load tokenizer from {:?}: {}", path, source_message),
            TokenizerError::EncodingFailed{ text, source_message } => write!(f, "Encoding failed for text '{}': {}", text, source_message),
            TokenizerError::DecodingFailed{ ids, source_message } => write!(f, "Decoding failed for IDs {:?}: {}", ids, source_message),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::Io(ref err) => Some(err),
            // Library error string is stored, so no deeper source.
            // If TokenizersLibraryError was boxed, it could be returned.
            TokenizerError::Library(_) => None, 
            TokenizerError::FailedToLoad { .. } => None, // source_message is String
            TokenizerError::EncodingFailed { .. } => None, // source_message is String
            TokenizerError::DecodingFailed { .. } => None, // source_message is String
        }
    }
}

// Implement From<std::io::Error> for TokenizerError
impl From<io::Error> for TokenizerError {
    fn from(err: io::Error) -> Self {
        TokenizerError::Io(err)
    }
}

// Implement From<tokenizers::Error> for TokenizerError
impl From<TokenizersLibraryError> for TokenizerError {
    fn from(err: TokenizersLibraryError) -> Self {
        TokenizerError::Library(err.to_string())
    }
}

/// A wrapper around the Hugging Face `tokenizers::Tokenizer` to provide a simplified API
/// for common tokenization tasks such as loading, encoding, decoding, and vocabulary size retrieval.
///
/// This struct holds an instance of `tokenizers::Tokenizer` and provides methods
/// that map to its core functionalities, along with custom error handling
/// defined by `TokenizerError`. It supports loading pre-configured tokenizers from a single
/// JSON file (via `new()`) or constructing specific tokenizers like GPT-2 from their
/// constituent vocabulary and merge files (via `from_gpt2_files()`).
///
/// ## Optional Debug Logging
/// This crate includes optional debug logging via the `log` crate. To enable it, compile
/// with the `tokenizer-debug-logs` feature:
///
/// ```bash
/// cargo build --features tokenizer-debug-logs
/// ```
///
/// You will also need to use a logger implementation (e.g., `env_logger`, `simple_logger`)
/// in your application to see the log output.
#[derive(Debug)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    /// Creates a new `TokenizerWrapper` by loading a tokenizer model from the specified file path.
    /// This method is suitable for loading tokenizer configurations that are fully defined
    /// in a single `tokenizer.json` file.
    ///
    /// # Parameters
    /// - `tokenizer_path`: A reference to a `Path` pointing to the tokenizer model file
    ///   (e.g., a `tokenizer.json` file).
    ///
    /// # Returns
    /// - `Ok(Self)`: A new `TokenizerWrapper` instance if loading the tokenizer model is successful.
    /// - `Err(TokenizerError::FailedToLoad)`: If the tokenizer model cannot be loaded.
    ///   The error includes the path and the underlying error message from the library.
    pub fn new(tokenizer_path: &Path) -> Result<Self, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!("Attempting to load tokenizer from path: {:?}", tokenizer_path);

        let tokenizer_instance = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| {
                #[cfg(feature = "tokenizer-debug-logs")]
                warn!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
                TokenizerError::FailedToLoad {
                    path: tokenizer_path.to_path_buf(),
                    source_message: e.to_string(),
                }
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Successfully loaded tokenizer from path: {:?}", tokenizer_path);
        Ok(Self { tokenizer: tokenizer_instance })
    }

    /// Creates a new `TokenizerWrapper` specifically configured for a GPT-2 style tokenizer
    /// from the given vocabulary and merges files.
    ///
    /// This method sets up a Byte Pair Encoding (BPE) model using the provided vocabulary
    /// and merge rules. It configures the tokenizer with components standard for GPT-2:
    /// - **Model**: BPE model with `<|endoftext|>` as the unknown token.
    /// - **Pre-tokenizer**: `ByteLevel` pre-tokenizer with `add_prefix_space: false` and `trim_offsets: true`.
    /// - **Decoder**: `ByteLevel` decoder.
    ///
    /// No explicit normalizer (e.g., NFC) or post-processor (e.g., for BOS/EOS tokens like `<|endoftext|>`)
    /// is added by this method. The `add_special_tokens` flag in the `encode` method might not automatically
    /// prepend/append BOS/EOS tokens unless the underlying model or a (not-set-here) post-processor handles it.
    /// For GPT-2, `<|endoftext|>` (ID 50256) often serves as the EOS token and sometimes as a BOS token,
    /// typically managed by the application or training process by including it in the input sequence.
    ///
    /// # Parameters
    /// - `vocab_path`: Path to the GPT-2 vocabulary JSON file (e.g., `gpt2-vocab.json`).
    ///   This file maps tokens to their IDs.
    /// - `merges_path`: Path to the GPT-2 merges text file (e.g., `merges.txt`).
    ///   This file defines the BPE merge rules.
    ///
    /// # Returns
    /// - `Ok(Self)`: A new `TokenizerWrapper` instance configured for GPT-2.
    /// - `Err(TokenizerError::FailedToLoad)`: If path conversion to string fails (e.g., invalid UTF-8 in paths)
    ///   or if the vocab/merges files cannot be read by the BPE builder.
    /// - `Err(TokenizerError::Library)`: If building the BPE model or configuring the tokenizer components fails
    ///   for other reasons (e.g., an invalid `unk_token` if it were not part of the vocab).
    pub fn from_gpt2_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Attempting to load GPT-2 tokenizer from vocab: {:?}, merges: {:?}",
            vocab_path, merges_path
        );

        let vocab_str = vocab_path.to_str().ok_or_else(|| TokenizerError::FailedToLoad {
            path: vocab_path.to_path_buf(),
            source_message: "Invalid vocab path (not valid UTF-8)".to_string()
        })?;
        let merges_str = merges_path.to_str().ok_or_else(|| TokenizerError::FailedToLoad {
            path: merges_path.to_path_buf(),
            source_message: "Invalid merges path (not valid UTF-8)".to_string()
        })?;

        let bpe_builder = BPE::builder()
            .files(vocab_str.to_string(), merges_str.to_string())
            .unk_token("<|endoftext|>".to_string());

        let bpe_model = bpe_builder.build()
            .map_err(|e| TokenizerError::Library(format!("Failed to build BPE model: {}", e)))?;

        let mut tokenizer = Tokenizer::new(bpe_model);

        // GPT-2 uses byte-level pre-tokenization without adding a prefix space by default.
        // ByteLevelPretokenizer::new(add_prefix_space: bool, trim_offsets: bool (use_regex in some versions))
        // For GPT-2, add_prefix_space is typically false. trim_offsets is true.
        let pre_tokenizer = ByteLevelPretokenizer::new(false, true, true);
        tokenizer.with_pre_tokenizer(pre_tokenizer);
        
        tokenizer.with_decoder(ByteLevelDecoder::default());

        #[cfg(feature = "tokenizer-debug-logs")]
        trace!(
            "Successfully configured GPT-2 style tokenizer. Vocab size: {}",
            tokenizer.get_vocab_size(true)
        );

        Ok(Self { tokenizer })
    }

    /// Creates a new `TokenizerWrapper` by loading a base GPT-2 tokenizer and then
    /// adding a list of custom symbolic tokens.
    ///
    /// This is useful for scenarios where a standard GPT-2 model needs to be augmented
    /// with specific symbols or control tokens (e.g., `"[USER]"`, `"[ASSISTANT]"`, `"[END_OF_TURN]"`).
    /// These symbols are added as special, single-word tokens that will not be split by
    /// the BPE model and will bypass some normalization steps.
    ///
    /// The base GPT-2 tokenizer is loaded from default paths:
    /// - Vocab: `resources/tokenizer_data/gpt2/gpt2-vocab.json`
    /// - Merges: `resources/tokenizer_data/gpt2/merges.txt`
    /// (Relative to `CARGO_MANIFEST_DIR`).
    ///
    /// After loading the base tokenizer, the provided `symbols_to_add` are registered.
    /// Their assigned token IDs can be retrieved using methods like `tokenizer.token_to_id(symbol)`.
    ///
    /// # Parameters
    /// - `symbols_to_add`: A slice of string slices, where each string is a new symbolic token
    ///   to be added to the GPT-2 tokenizer.
    ///
    /// # Returns
    /// - `Ok(Self)`: A new `TokenizerWrapper` instance with the base GPT-2 tokenizer augmented
    ///   with the specified symbolic tokens.
    /// - `Err(TokenizerError)`: If loading the base GPT-2 tokenizer fails, or if adding
    ///   the new symbolic tokens fails. Possible error types include `TokenizerError::FailedToLoad`
    ///   (if vocab/merges paths are problematic or files are missing/corrupt) or
    ///   `TokenizerError::Library` (if BPE model building or adding tokens fails).
    pub fn from_symbolic_gpt2(symbols_to_add: &[&str]) -> Result<Self, TokenizerError> {
        let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let vocab_path = base_dir.join("resources/tokenizer_data/gpt2/gpt2-vocab.json");
        let merges_path = base_dir.join("resources/tokenizer_data/gpt2/merges.txt");

        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Creating symbolic GPT-2 tokenizer. Base vocab: {:?}, base merges: {:?}, symbols to add: {:?}",
            vocab_path, merges_path, symbols_to_add
        );

        if !vocab_path.exists() {
            return Err(TokenizerError::FailedToLoad { path: vocab_path, source_message: "GPT-2 vocab file not found.".to_string() });
        }
        if !merges_path.exists() {
            return Err(TokenizerError::FailedToLoad { path: merges_path, source_message: "GPT-2 merges file not found.".to_string() });
        }

        let mut tokenizer_wrapper = Self::from_gpt2_files(&vocab_path, &merges_path)?;

        if !symbols_to_add.is_empty() {
            let added_tokens: Vec<AddedToken> = symbols_to_add
                .iter()
                .map(|s| AddedToken::from(*s, true).special(true)) // `true` for single_word, .special(true) to ensure it's treated as special
                .collect();
            
            #[cfg(feature = "tokenizer-debug-logs")]
            debug!("Adding symbolic tokens to GPT-2 base: {:?}", added_tokens);

            match tokenizer_wrapper.add_new_tokens(&added_tokens) {
                Ok(num_added) => {
                    #[cfg(feature = "tokenizer-debug-logs")]
                    trace!("Successfully added {} symbolic tokens.", num_added);
                }
                Err(e) => {
                    #[cfg(feature = "tokenizer-debug-logs")]
                    warn!("Failed to add symbolic tokens: {}", e);
                    return Err(e); // Propagate the error
                }
            }
        }
        
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Successfully created symbolic GPT-2 tokenizer.");
        Ok(tokenizer_wrapper)
    }


    /// Encodes a given text string into a sequence of token IDs.
    ///
    /// # Parameters
    /// - `text`: The text string to be encoded.
    /// - `add_special_tokens`: A boolean indicating whether to include special tokens
    ///   (e.g., `[CLS]`, `[SEP]`) in the encoded output, as defined by the tokenizer model's
    ///   configuration.
    ///
    /// # Returns
    /// - `Ok(Vec<u32>)`: A vector of `u32` token IDs representing the encoded text.
    /// - `Err(TokenizerError::EncodingFailed)`: If encoding fails.
    ///   The error includes the input text and the underlying error message from the library.
    ///
    /// # Parameters
    /// - `text`: The text string to be encoded.
    /// - `add_special_tokens`: A boolean indicating whether to include special tokens.
    /// - `options`: An optional `EncodeOptions` struct to specify truncation and padding.
    ///   If `None`, default encoding behavior (no specific truncation or padding for this call) is used.
    pub fn encode(&self, text: &str, add_special_tokens: bool, options: Option<EncodeOptions>) -> Result<Vec<u32>, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Encoding text: '{}', add_special_tokens: {}, options: {:?}",
            text, add_special_tokens, options
        );

        let mut tokenizer_instance = self.tokenizer.clone(); // Clone to apply per-call settings

        let (truncation_params, padding_params) = match options {
            Some(opts) => (opts.truncation, opts.padding),
            None => (None, None),
        };

        #[cfg(feature = "tokenizer-debug-logs")]
        if let Some(ref params) = truncation_params {
            trace!("Applying truncation parameters: {:?}", params);
        }
        // Assuming with_truncation exists and returns Result
        tokenizer_instance.with_truncation(truncation_params.clone())
            .map_err(|e| TokenizerError::EncodingFailed {
                text: format!("Failed to set truncation for text '{}'", if text.len() > 50 { &text[..50] } else { text }),
                source_message: e.to_string(),
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        if let Some(ref params) = padding_params {
            trace!("Applying padding parameters: {:?}", params);
        }
        // with_padding returns &mut Self, so no map_err or ?
        tokenizer_instance.with_padding(padding_params.clone());

        let encoding_result = tokenizer_instance.encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::EncodingFailed {
                text: text.to_string(), // Full text for this error context
                source_message: e.to_string(),
            })?;
        
        let encoded_ids = encoding_result.get_ids().to_vec();
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Encoded IDs: {:?}", encoded_ids);
        Ok(encoded_ids)
    }

    /// Decodes a sequence of token IDs back into a text string.
    ///
    /// # Parameters
    /// - `ids`: A slice of `u32` token IDs to be decoded.
    /// - `skip_special_tokens`: A boolean indicating whether to remove special tokens
    ///   (e.g., `[CLS]`, `[SEP]`, `[PAD]`) from the decoded string. This is based on the
    ///   tokenizer model's definition of special tokens.
    ///
    /// # Returns
    /// - `Ok(String)`: The decoded text string.
    /// - `Err(TokenizerError::DecodingFailed)`: If decoding fails.
    ///   The error includes the token IDs and the underlying error message from the library.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!(
            "Decoding IDs: {:?}, skip_special_tokens: {}",
            ids, skip_special_tokens
        );

        let decoded_string = self.tokenizer.decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::DecodingFailed {
                ids: ids.to_vec(),
                source_message: e.to_string(),
            })?;

        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Decoded text: '{}'", decoded_string);
        Ok(decoded_string)
    }

    /// Retrieves the vocabulary size of the loaded tokenizer.
    ///
    /// # Parameters
    /// - `with_added_tokens`: A boolean that determines whether to include "added tokens"
    ///   (tokens not part of the core model vocabulary but added for specific purposes,
    ///   like special control tokens) in the count.
    ///   - `true`: Includes added tokens in the vocabulary size.
    ///   - `false`: Returns the size of the base vocabulary, excluding added tokens
    ///     (unless those tokens are also part of the base vocabulary itself).
    ///   The exact interpretation may depend on the tokenizer's configuration.
    ///
    /// # Returns
    /// - `u32`: The size of the vocabulary as a `u32` integer.
    // The duplicated get_vocab_size was an artifact from a previous incorrect merge.
    // This is the correct single definition.
    pub fn get_vocab_size(&self) -> u32 {
        // This method is simple enough that perhaps only a trace log is needed.
        // The parameter `with_added_tokens` was part of a previous iteration but removed.
        // The current signature takes no parameters and defaults to `true`.
        let vocab_size = self.tokenizer.get_vocab_size(true) as u32;
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!(
            "Getting vocab size (with_added_tokens=true). Returned size: {}",
            vocab_size
        );
        vocab_size
    }

    /// Adds new tokens to the tokenizer's vocabulary at runtime.
    ///
    /// This method allows extending the existing vocabulary with new tokens.
    /// The added tokens are typically treated as special tokens or whole words,
    /// depending on their configuration and the underlying tokenizer model.
    /// For example, to add a token that should be treated as a single unit and not
    /// be split by the model, you might use `AddedToken::from("new_word", true).single_word(true)`.
    ///
    /// Note: This modifies the internal tokenizer instance (`&mut self`).
    ///
    /// # Parameters
    /// - `tokens`: A slice of `AddedToken` structs representing the new tokens to add.
    ///
    /// # Returns
    /// - `Ok(usize)`: The number of tokens effectively added to the vocabulary.
    /// - `Err(TokenizerError::Library)`: If adding tokens fails (e.g., due to internal tokenizer issues).
    pub fn add_new_tokens(&mut self, tokens: &[AddedToken]) -> Result<usize, TokenizerError> {
        #[cfg(feature = "tokenizer-debug-logs")]
        debug!("Adding new tokens: {:?}", tokens);

        let num_added = self.tokenizer.add_tokens(tokens);
        
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Successfully added {} tokens. New vocab size: {}", num_added, self.tokenizer.get_vocab_size(true));
        Ok(num_added)
    }

    /// Retrieves the numerical ID of a given token string (symbol) from the tokenizer's vocabulary.
    ///
    /// This method can be used to find the ID for any token known to the tokenizer,
    /// including standard vocabulary tokens and custom symbols that might have been
    /// added via methods like `add_new_tokens` or constructors like `from_symbolic_gpt2`.
    ///
    /// # Parameters
    /// - `symbol`: The token string (e.g., "hello", "<|endoftext|>", `"[USER_PROMPT]"`) whose ID is to be retrieved.
    ///
    /// # Returns
    /// - `Some(u32)`: If the token is found in the vocabulary, containing its numerical ID.
    /// - `None`: If the token is not found in the vocabulary.
    pub fn get_symbol_id(&self, symbol: &str) -> Option<u32> {
        let result = self.tokenizer.token_to_id(symbol);
        #[cfg(feature = "tokenizer-debug-logs")]
        trace!("Looking up ID for symbol: '{}', Found: {:?}", symbol, result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write; // For NamedTempFile::write_all
    use tempfile::NamedTempFile;
    use tokenizers::AddedToken; // Required for add_new_tokens test

    // Define the dummy tokenizer JSON content
    // This structure is based on typical Hugging Face tokenizer files.
    const DUMMY_TOKENIZER_JSON: &str = "{\n\
      \"model\": {\n\
        \"type\": \"BPE\",\n\
        \"vocab\": {\n\
          \"[PAD]\": 0,\n\
          \"[UNK]\": 1,\n\
          \"[CLS]\": 2,\n\
          \"[SEP]\": 3,\n\
          \"[MASK]\": 4,\n\
          \"hello\": 5,\n\
          \"world\": 6,\n\
          \"##d\": 7,\n\
          \"Ġ\": 8,\n\
          \"Ġhello\": 9,\n\
          \"Ġworld\": 10,\n\
          \"Ġ##d\" : 11\n\
        },\n\
        \"merges\": [\n\
          \"Ġ h\",\n\
          \"Ġ w\",\n\
          \"h e\",\n\
          \"l l\",\n\
          \"o w\",\n\
          \"r l\",\n\
          \"l d\",\n\
          \"Ġ ##\",\n\
          \"## d\"\n\
        ]\n\
      },\n\
      \"normalizer\": {\n\
        \"type\": \"BertNormalizer\",\n\
        \"lowercase\": true,\n\
        \"strip_accents\": true,\n\
        \"clean_text\": true\n\
      },\n\
      \"pre_tokenizer\": {\n\
        \"type\": \"BertPreTokenizer\"\n\
      },\n\
      \"post_processor\": {\n\
        \"type\": \"BertProcessing\",\n\
        \"sep\": [\"[SEP]\", 3],\n\
        \"cls\": [\"[CLS]\", 2]\n\
      },\n\
      \"decoder\": {\n\
        \"type\": \"BPEDecoder\",\n\
        \"suffix\": \"Ġ\"\n\
      }\n\
    }";

    // This helper function is defined at the `tests` module level to be shared.
    // It ensures the NamedTempFile lives as long as the TokenizerWrapper instance.
    fn setup_temp_tokenizer_file_and_wrapper(json_content: &str) -> (NamedTempFile, TokenizerWrapper) {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file for test");
        temp_file.write_all(json_content.as_bytes()).expect("Failed to write to temporary test file");
        temp_file.flush().expect("Failed to flush temporary test file");
        let wrapper = TokenizerWrapper::new(temp_file.path())
            .expect("Failed to load TokenizerWrapper from temporary test file");
        (temp_file, wrapper)
    }

    #[test]
    fn test_tokenizer_new_load_fails_for_nonexistent_file() {
        let non_existent_path = Path::new("non_existent_tokenizer.json");
        let result = TokenizerWrapper::new(non_existent_path);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::FailedToLoad{ path, source_message }) => {
                println!("Load non-existent file error (FailedToLoad): Path: {:?}, Source: {}", path, source_message);
                assert_eq!(path, non_existent_path.to_path_buf());
                // The source_message from tokenizers::Error for a non-existent file is usually specific.
                assert!(source_message.contains("No such file or directory") || source_message.contains("failed to read file"));
            }
            other_err => panic!("Expected FailedToLoad error variant, got {:?}", other_err),
        }
    }

    #[test]
    fn test_tokenizer_new_load_succeeds_and_get_vocab_size() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON); 
        // No need to assert wrapper_result.is_ok(), as setup_..._and_wrapper panics on failure.
        
        // Vocab from DUMMY_TOKENIZER_JSON:
        // "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
        // "hello": 5, "world": 6, "##d": 7, "Ġ": 8,
        // "Ġhello": 9, "Ġworld": 10, "Ġ##d" : 11
        // Total 12 tokens in the "model.vocab".
        // `get_vocab_size()` calls `self.tokenizer.get_vocab_size(true)`, which includes added tokens.
        // In this JSON, CLS and SEP are also defined in the main vocab and used by post_processor.
        // So, the vocab size should be the number of entries in "model.vocab".
        assert_eq!(wrapper.get_vocab_size(), 12, "Vocab size mismatch for dummy tokenizer.");
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

        let text_to_encode = "hello world";
        // Pass None for options to keep original test behavior
        let encode_result = wrapper.encode(text_to_encode, false, None); 
        assert!(encode_result.is_ok(), "Encoding failed: {:?}", encode_result.err().map(|e| e.to_string()));
        let encoded_ids = encode_result.unwrap();
        
        // With DUMMY_TOKENIZER_JSON:
        // Input: "hello world"
        // BertNormalizer (lowercase: true): "hello world"
        // BertPreTokenizer: Splits into words: "hello", "world"
        // BPE Model:
        //   "hello" is in vocab -> 5
        //   For "world", BertPreTokenizer gives "world". The BPE model needs "Ġworld" for ID 10.
        //   The pre_tokenizer should handle the space: "hello", "Ġworld" if it's not the first word.
        //   However, `encode` on `Tokenizer` with `BertPreTokenizer` usually handles space prefixing.
        //   So, "hello" -> 5. "world" (after first word) -> "Ġworld" -> 10.
        // Expected: [5, 10]
        assert_eq!(encoded_ids, vec![5, 10], "Encoded IDs do not match expected for 'hello world'.");

        let decode_result = wrapper.decode(&encoded_ids, true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding failed: {:?}", decode_result.err().map(|e| e.to_string()));
        let decoded_text = decode_result.unwrap();

        // The HuggingFace tokenizers library often lowercases by default with BertNormalizer,
        // and might not perfectly reconstruct the original casing if not configured.
        // The dummy tokenizer uses BertNormalizer with lowercase: true.
        assert_eq!(decoded_text, "hello world", "Decoded text does not match original (post-normalization).");

        // Test with special tokens
        let text_for_special_tokens = "hello world"; // User input part
        // Pass None for options
        let encoded_special_result = wrapper.encode(text_for_special_tokens, true, None); 
        assert!(encoded_special_result.is_ok(), "Encoding special tokens failed: {:?}", encoded_special_result.err().map(|e| e.to_string()));
        let encoded_special_ids = encoded_special_result.unwrap();

        // Input: "hello world", add_special_tokens=true
        // Normalized: "hello world"
        // Pre-tokenized by BertPreTokenizer: "hello", "world"
        // BPE encoding of parts: "hello" -> 5, "world" (becomes "Ġworld" due to space) -> 10. Intermediate: [5, 10]
        // Post-processor (BertProcessing): Adds [CLS] (ID 2) at start, [SEP] (ID 3) at end.
        // Expected: [2, 5, 10, 3]
        assert_eq!(encoded_special_ids, vec![2, 5, 10, 3], "Encoded IDs with special tokens do not match.");

        // Decode skipping special tokens
        let decoded_skip_special = wrapper.decode(&encoded_special_ids, true).unwrap();
        assert_eq!(decoded_skip_special, "hello world");
        
        // Decode without skipping special tokens
        let decoded_no_skip_special = wrapper.decode(&encoded_special_ids, false).unwrap();
        assert_eq!(decoded_no_skip_special, "[CLS] hello world [SEP]");
    }

    #[test]
    fn test_encode_error_handling() {
        // This test assumes that the underlying tokenizer's encode method can fail
        // For example, if it encounters an unhandled situation or internal limit.
        // Since the dummy tokenizer is very simple, it's hard to make it fail reliably
        // without specific knowledge of `tokenizers::Tokenizer` internals that cause errors
        // beyond simple string processing.
        // We'll simulate a failure by trying to encode a ridiculously long string,
        // though this might just be slow rather than erroring with the default settings.
        // A more robust test would mock the Tokenizer trait if we were using one, or
        // create a specific tokenizer file known to cause encoding issues for certain inputs.

        // For now, this test is more of a placeholder for error *type* checking
        // if an actual encoding error were to occur.
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        
        // Let's assume a hypothetical scenario where encoding an empty string after certain ops might fail
        // (this is not typical for `encode` which usually returns empty for empty).
        // Or if a specific (malformed from user perspective) sequence of tokens might be problematic.
        // The current `tokenizers` lib is quite robust.
        // Instead, we'll check that if an error *were* to occur, its type is correct.
        // We can't easily make `tokenizer.encode()` fail with the current setup.
        // So, this test primarily ensures the error propagation *would* work.
        // No actual error expected here for valid inputs.
        // Pass None for options
        let result = wrapper.encode("test", false, None); 
        if let Err(e) = result {
            match e {
                TokenizerError::EncodingFailed{ text: _, source_message: _ } => { /* This is what we'd expect if it failed */ }
                // If for some reason it was a Library error (e.g. direct ? from another source)
                TokenizerError::Library(_) => { /* Potentially acceptable if source_message is good */ }
                _ => panic!("Unexpected error type for encoding failure: {:?}", e),
            }
        }
    }

    #[test]
    fn test_decode_error_handling() {
        // Similar to encode, making `decode` fail typically means providing invalid IDs
        // that are out of bounds of the vocabulary.
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);

        let invalid_ids = vec![u32::MAX]; // An ID that's almost certainly not in the vocab
        let result = wrapper.decode(&invalid_ids, false);
        assert!(result.is_err());
        match result {
            Err(TokenizerError::DecodingFailed{ ids, source_message }) => {
                println!("Decoding failed as expected (DecodingFailed): IDs: {:?}, Source: {}", ids, source_message);
                assert_eq!(ids, invalid_ids); // Removed &
                // The source_message from tokenizers::Error for invalid ID might be like "TokenId `X` out of vocabulary bounds"
                assert!(source_message.contains("out of vocabulary bounds") || source_message.contains("invalid id"));
            }
            other_err => panic!("Expected DecodingFailed error variant, got {:?}", other_err),
        }
    }
    
    // Optional: Clean up the dummy tokenizer file after all tests in this module if it's not version controlled.
    // This would require a custom test runner or a teardown mechanism, which is complex.
    // For now, manual cleanup or versioning the test_tokenizer.json is assumed.

    // Added DUMMY_SPECIAL_TOKEN_ID for tests - assuming <|endoftext|> is 50256 or similar
    // This should align with what the dummy tokenizer might represent for a generic special token
    // if it's not one of the BERT specific ones. For the old get_dummy_tokenizer_path tests,
    // they were using a different tokenizer.json.
    // The DUMMY_TOKENIZER_JSON does not have <|endoftext|>.
    // Let's assume [PAD] (0) or [UNK] (1) as a placeholder for "special" in these tests if not BERT ones.
    // Or better, use a token that *is* in DUMMY_TOKENIZER_JSON like [MASK] (4)
    // For "test_encode_only_special_tokens_string" and "test_decode_only_special_tokens_ids"
    // they were testing with "<|endoftext|>" and DUMMY_SPECIAL_TOKEN_ID.
    // This implies that `get_dummy_tokenizer_path` was pointing to a tokenizer with that token.
    // The current DUMMY_TOKENIZER_JSON does not have it.
    // For the purpose of fixing the call sites, I will use a token that IS in DUMMY_TOKENIZER_JSON.
    // Let's use "[MASK]" which has ID 4.
    const DUMMY_SPECIAL_TOKEN_ID_FOR_TESTS: u32 = 4; // Using [MASK] ID as a stand-in
    const DUMMY_SPECIAL_TOKEN_STR_FOR_TESTS: &str = "[MASK]"; // Using [MASK] string

    #[test]
    fn test_encode_empty_string() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        let encode_result = wrapper.encode("", false, None);
        assert!(encode_result.is_ok(), "Encoding empty string failed: {:?}", encode_result.err());
        assert!(encode_result.unwrap().is_empty(), "Encoding an empty string should result in an empty Vec<u32>");
    }

    #[test]
    fn test_decode_empty_ids() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        let decode_result = wrapper.decode(&[], true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding empty ID list failed: {:?}", decode_result.err());
        assert!(decode_result.unwrap().is_empty(), "Decoding an empty Vec<u32> should result in an empty String");
    }

    #[test]
    fn test_encode_oov_string() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        let text_oov = "xyz"; // These characters are not in DUMMY_TOKENIZER_JSON's vocab
        let encode_result = wrapper.encode(text_oov, false, None);
        assert!(encode_result.is_ok(), "Encoding OOV string failed: {:?}", encode_result.err());
        let encoded_ids = encode_result.unwrap();
        // DUMMY_TOKENIZER_JSON uses BertNormalizer (lowercase) and BertPreTokenizer.
        // "xyz" will likely be tokenized to "[UNK]" (ID 1) by the BPE model if "x", "y", "z" are not in vocab.
        // The dummy vocab has: "[UNK]": 1
        assert_eq!(encoded_ids, vec![1], "Encoding OOV string 'xyz' should result in UNK token ID.");

        // Test decoding this UNK token
        let decode_result = wrapper.decode(&encoded_ids, true); // skip_special_tokens = true
        assert!(decode_result.is_ok(), "Decoding UNK token failed: {:?}", decode_result.err());
        // In DUMMY_TOKENIZER_JSON, "[UNK]" is not specifically marked as a special token to be skipped
        // by the decoder's `skip_special_tokens` flag in the same way [CLS], [SEP] are by BertProcessing.
        // However, the "vocab" entries themselves don't define "special" status for skipping.
        // The `tokenizers` library's `decode` with `skip_special_tokens=true` typically skips tokens
        // that were added as `special=true` or are handled by a post-processor.
        // The [UNK] token itself, when decoded, might result in an empty string or the UNK string itself.
        // For DUMMY_TOKENIZER_JSON, let's check what it decodes to.
        // The `BPEDecoder` in DUMMY_TOKENIZER_JSON might decode ID 1 back to "[UNK]".
        // If skip_special_tokens is true, and [UNK] is treated as special, it would be empty.
        // Given the setup, it's safer to assume it might decode to "[UNK]" or "" depending on interpretation.
        // The original test expected "". Let's stick to that if [UNK] is implicitly special.
        // From `tokenizers` crate docs: "special tokens are cleaned up".
        // Let's assume [UNK] is considered special enough to be cleaned.
        assert_eq!(decode_result.unwrap(), "[UNK]", "Decoding UNK token with skip_special_tokens=true did not produce expected string. It should be '[UNK]' as it is not skipped by default by the BPE decoder unless it's part of post-processor's list of special tokens to remove.");
    }

    #[test]
    fn test_encode_only_special_tokens_string() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        // Using [MASK] as defined by DUMMY_SPECIAL_TOKEN_STR_FOR_TESTS
        let text_special = DUMMY_SPECIAL_TOKEN_STR_FOR_TESTS;
        let encode_result = wrapper.encode(text_special, true, None); // add_special_tokens = true
        assert!(encode_result.is_ok(), "Encoding string of only special tokens failed: {:?}", encode_result.err());
        // BertProcessing post-processor in DUMMY_TOKENIZER_JSON adds [CLS] and [SEP].
        // Input: "[MASK]", add_special_tokens=true
        // Normalized: "[mask]" (due to BertNormalizer lowercase:true)
        // Pre-tokenized: "[mask]"
        // BPE model: "[MASK]" is ID 4.
        // Post-processor: adds [CLS] (2) at start, [SEP] (3) at end.
        // Expected: [2, 4, 3]
        assert_eq!(encode_result.unwrap(), vec![2, DUMMY_SPECIAL_TOKEN_ID_FOR_TESTS, 3], "Encoding string of special tokens produced incorrect IDs.");
    }

    #[test]
    fn test_decode_only_special_tokens_ids() {
        let (_temp_file, wrapper) = setup_temp_tokenizer_file_and_wrapper(DUMMY_TOKENIZER_JSON);
        // Using [MASK] ID as defined by DUMMY_SPECIAL_TOKEN_ID_FOR_TESTS
        let special_ids = vec![DUMMY_SPECIAL_TOKEN_ID_FOR_TESTS];
        
        // Decode skipping special tokens
        let decode_skip_result = wrapper.decode(&special_ids, true); // skip_special_tokens = true
        assert!(decode_skip_result.is_ok(), "Decoding special IDs (skip=true) failed: {:?}", decode_skip_result.err());
        // DUMMY_TOKENIZER_JSON has BertNormalizer (lowercase) and BPEDecoder.
        // [MASK] (ID 4) when decoded with skip_special_tokens=true.
        // The BertProcessing post-processor defines [CLS] and [SEP] as special for removal.
        // Individual tokens from vocab like [MASK], [UNK], [PAD] are not automatically skipped by
        // `skip_special_tokens=true` unless they are part of the template output of a post-processor
        // (like [CLS] text [SEP]) and then removed.
        // So, decoding just ID 4 ([MASK]) with skip_special_tokens=true should still yield "[MASK]".
        assert_eq!(decode_skip_result.unwrap(), DUMMY_SPECIAL_TOKEN_STR_FOR_TESTS, "Decoding special ID [MASK] (skip=true) should result in '[MASK]'.");

        // Decode without skipping special tokens
        let decode_no_skip_result = wrapper.decode(&special_ids, false); // skip_special_tokens = false
        assert!(decode_no_skip_result.is_ok(), "Decoding special IDs (skip=false) failed: {:?}", decode_no_skip_result.err());
        assert_eq!(decode_no_skip_result.unwrap(), DUMMY_SPECIAL_TOKEN_STR_FOR_TESTS, "Decoding special ID [MASK] (skip=false) did not produce expected string.");
    }
}
