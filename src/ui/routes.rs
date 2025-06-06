//! # Web Server Routes for File Uploader UI
//!
//! This module defines the Actix web server routes and handlers for the file
//! uploading and inspection user interface. It includes routes to serve the main
//! HTML page and to handle file uploads.

const VOCAB_PREVIEW_COUNT: usize = 30;

use actix_web::{web, App, HttpServer, Responder, HttpResponse, Error};
use actix_files::NamedFile;
use std::path::PathBuf; // Though PathBuf is imported, it's not explicitly used in the provided snippet. Will keep for now.
use actix_multipart::Multipart;
use futures_util::TryStreamExt;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use html_escape;

/// Serves the main HTML page (`index.html`) for the file uploader UI.
///
/// This function handles GET requests to the root path (`/`).
/// It asynchronously opens and returns the `index.html` file.
pub async fn index() -> impl Responder { // Made public
    NamedFile::open_async("./src/ui/index.html")
        .await
        .expect("index.html should be present in src/ui/") // .unwrap() can panic, added expect for clarity.
}

/// Handles multipart file uploads from the UI.
///
/// This function processes POST requests to the `/upload` path. It expects
/// `multipart/form-data` containing one or more files.
///
/// For each uploaded file, it attempts to:
/// 1. Identify if it's a `.safetensors` model weights file or a `tokenizer.json` file.
/// 2. Parse the file to extract relevant metadata:
///    - For `.safetensors`: Tensor names and their shapes.
///    - For `tokenizer.json`: A preview of the JSON content.
/// 3. Collect any processing errors or skip unsupported files.
///
/// # Memory Considerations:
/// - Each file's content is currently read entirely into a `Vec<u8>` in memory.
///   For very large files (especially `.safetensors`), this can lead to high peak RAM usage.
///   More advanced solutions like memory-mapping temporary files could mitigate this but add complexity.
/// - The `safetensors` crate itself is efficient for metadata extraction once the data is in memory,
///   as it uses views into the byte slice.
/// - `tokenizer.json` files are typically small, making the full read and `serde_json::Value` DOM
///   parsing generally acceptable.
///
/// # Returns
/// An `HttpResponse` which is one of:
/// - **200 OK**: If files were processed (even if some were skipped). The body contains HTML
///   displaying information about successfully processed files and/or messages about skipped files.
///   If no compatible files are processed, a relevant message is shown.
/// - **400 Bad Request**: If any file processing results in a critical error (e.g., corrupted
///   SafeTensors file, invalid JSON). The body contains HTML detailing these errors,
///   alongside information about any files that might have been processed successfully before the error.
/// - Other Actix web error types might be returned by the framework itself on lower-level issues.
pub async fn upload_files(mut payload: Multipart) -> Result<HttpResponse, Error> { // Made public
    let mut uploaded_files_info = HashMap::new();
    let mut processing_errors = Vec::new();
    // Flag to track if any user-facing error (parsing, wrong type) occurred.
    // This helps decide if the overall response should be OK or Bad Request.
    let mut any_critical_error_occurred = false;

    while let Some(item_result) = payload.try_next().await {
        let mut field = match item_result {
            Ok(field) => field,
            Err(e) => {
                processing_errors.push(format!("Error reading multipart stream: {}", e));
                any_critical_error_occurred = true; // Error in stream reading is critical
                continue;
            }
        };

        let content_disposition = field.content_disposition();
        let filename = content_disposition.get_filename().unwrap_or("unknown_file").to_string();

        // Memory consideration: The entire file for this field is read into `file_bytes`.
        // (See previous comments on this topic)
        let mut file_bytes = Vec::new();
        let mut field_chunk_error = false;
        while let Some(chunk_result) = field.try_next().await {
            match chunk_result {
                Ok(chunk) => file_bytes.extend_from_slice(&chunk),
                Err(e) => {
                    let err_msg = format!("Error reading data for file '{}': {}", filename, e);
                    processing_errors.push(err_msg);
                    any_critical_error_occurred = true; // Error reading chunks is critical
                    field_chunk_error = true;
                    break;
                }
            }
        }

        if field_chunk_error {
            continue;
        }

        // Process the collected file bytes using the helper function
        match process_single_file(&filename, file_bytes).await {
            Ok((processed_filename, info_map)) => {
                // Check if it was a skip, which is reported as an Err by process_single_file if it adds to processing_errors.
                // This logic is now simplified: Ok means successfully processed.
                uploaded_files_info.insert(processed_filename, info_map);
            }
            Err(err_msg) => {
                // Errors from process_single_file can be parsing errors or "skipped file" messages.
                // We treat any such message as something to report to the user.
                // If the error indicates a true parsing failure vs. a skip, that could influence any_critical_error_occurred.
                // The current process_single_file returns Err for skips too.
                if err_msg.contains("Invalid") || err_msg.contains("corrupted") { // Heuristic for critical errors
                    any_critical_error_occurred = true;
                }
                processing_errors.push(err_msg);
            }
        }
    }

    // Determine overall response status.
    // If there were critical errors (parsing, stream read), it should be a Bad Request.
    // If only skips occurred, this might also be a Bad Request based on current logic,
    // or could be an OK with a list of issues.
    // The `any_critical_error_occurred` helps distinguish.
    // However, the current HTML generation logic for errors simply lists all `processing_errors`.
    // If `processing_errors` is not empty, it returns BadRequest.
    // This behavior is kept for now.
    // it should be a Bad Request. If files were just skipped (unsupported type),
    // it's more of a partial success, so OK status might still be appropriate,
    // but the message should clearly indicate skipped files.
    // The current logic pushes skipped messages to `processing_errors` and then returns BadRequest
    // if `processing_errors` is not empty. This means skipped files also lead to BadRequest.
    // This can be adjusted if skipped files should not cause a "request failure" status.
    // For now, any entry in `processing_errors` (including skips) makes it a "Bad Request" effectively.

    if !processing_errors.is_empty() {
        let mut error_html_response = String::new();
        error_html_response.push_str("<div class=\"error-message\">");
        error_html_response.push_str("<h2>File Processing Issues:</h2><ul>");
        for err in &processing_errors { // Borrow processing_errors
            error_html_response.push_str(&format!("<li>{}</li>", err));
        }
        error_html_response.push_str("</ul></div>");

        if !uploaded_files_info.is_empty() {
             error_html_response.push_str("<hr><h3>Successfully Processed Files (if any):</h3>");
        }
        // Also include info about successfully processed files if any
        // This part is similar to the success case but appended to the error message
         for (filename, info) in &uploaded_files_info {
            error_html_response.push_str(&format!("<h4>{}</h4>", filename));
            if info.get("_type").map_or(false, |t| t == "safetensor") {
                error_html_response.push_str("<ul>");
                let mut tensor_names = info.keys()
                    .filter_map(|k| k.split('.').next())
                    .filter(|&k| k != "_type")
                    .collect::<Vec<_>>();
                tensor_names.sort();
                tensor_names.dedup();

                for name in tensor_names {
                    let shape = info.get(&format!("{}.shape", name)).unwrap_or(&"N/A".to_string());
                    let dtype = info.get(&format!("{}.dtype", name)).unwrap_or(&"N/A".to_string());
                    let preview = info.get(&format!("{}.preview", name)).unwrap_or(&"N/A".to_string());
                    // Basic HTML escaping for attribute values, especially preview
                    let preview_escaped = html_escape::encode_double_quoted_attribute(preview);
                    let shape_escaped = html_escape::encode_double_quoted_attribute(shape);
                    let tensor_id = html_escape::encode_double_quoted_attribute(&format!("{}::{}", filename, name));

                    error_html_response.push_str(&format!(
                        "<li data-dtype=\"{}\" data-preview=\"{}\" data-shape=\"{}\" data-tensor-id=\"{}\">"
                        + "<input type=\"checkbox\" name=\"tensor_compare_select\" value=\"{}\"> "
                        + "{}: {}</li>",
                        dtype, preview_escaped, shape_escaped, tensor_id, // tensor_id for data attribute
                        tensor_id, // tensor_id for checkbox value
                        name, shape
                    ));
                }
                error_html_response.push_str("</ul>");
            } else if info.get("_type").map_or(false, |t| t == "tokenizer") {
                let mut pre_attributes = String::new();
                if let Some(vocab_json) = info.get("vocab_preview_json") {
                    pre_attributes.push_str(&format!(" data-vocab-preview='{}'", html_escape::encode_double_quoted_attribute(vocab_json)));
                }
                error_html_response.push_str(&format!("<pre{}>", pre_attributes));
                if let Some(content_preview) = info.get("content_preview") {
                    error_html_response.push_str(&content_preview.escape_default().to_string());
                }
                error_html_response.push_str("</pre>");
            }
        }
        return Ok(HttpResponse::BadRequest().content_type("text/html").body(error_html_response));
    }

    let mut html_response = String::new();
    if uploaded_files_info.is_empty() {
        // This case might be hit if only non-supported files were uploaded, leading to processing_errors
        // but no items in uploaded_files_info. The error block above handles showing those errors.
        // If processing_errors is also empty, it means no files were uploaded or they were empty.
        if processing_errors.is_empty() { // Only show this if there were no errors reported either
            html_response.push_str("<p>No files were uploaded or processed.</p>");
        }
    } else {
        html_response.push_str("<h2>Uploaded File Information:</h2>");
        for (filename, info) in uploaded_files_info {
            html_response.push_str(&format!("<h3>{}</h3>", filename));
            if info.get("_type").map_or(false, |t| t == "safetensor") {
                html_response.push_str("<ul>");
                // Extract tensor names, sort for consistent order
                let mut tensor_names = info.keys()
                    .filter_map(|k| k.split('.').next())
                    .filter(|&k| k != "_type") // Exclude the internal type hint
                    .collect::<Vec<_>>();
                tensor_names.sort();
                tensor_names.dedup();

                for name in tensor_names {
                    let shape = info.get(&format!("{}.shape", name)).unwrap_or(&"N/A".to_string());
                    let dtype = info.get(&format!("{}.dtype", name)).unwrap_or(&"N/A".to_string());
                    let preview = info.get(&format!("{}.preview", name)).unwrap_or(&"N/A".to_string());
                    // Basic HTML escaping for attribute values, especially preview
                    let preview_escaped = html_escape::encode_double_quoted_attribute(preview);
                    let shape_escaped = html_escape::encode_double_quoted_attribute(shape);
                    let tensor_id = html_escape::encode_double_quoted_attribute(&format!("{}::{}", filename, name));

                    html_response.push_str(&format!(
                        "<li data-dtype=\"{}\" data-preview=\"{}\" data-shape=\"{}\" data-tensor-id=\"{}\">"
                        + "<input type=\"checkbox\" name=\"tensor_compare_select\" value=\"{}\"> "
                        + "{}: {}</li>",
                        dtype, preview_escaped, shape_escaped, tensor_id, // tensor_id for data attribute
                        tensor_id, // tensor_id for checkbox value
                        name, shape
                    ));
                }
                html_response.push_str("</ul>");
            } else if info.get("_type").map_or(false, |t| t == "tokenizer") {
                let mut pre_attributes = String::new();
                if let Some(vocab_json) = info.get("vocab_preview_json") {
                    pre_attributes.push_str(&format!(" data-vocab-preview='{}'", html_escape::encode_double_quoted_attribute(vocab_json)));
                }
                html_response.push_str(&format!("<pre{}>", pre_attributes));
                if let Some(content_preview) = info.get("content_preview") {
                    html_response.push_str(&content_preview.escape_default().to_string());
                }
                html_response.push_str("</pre>");
            }
        }
    }
    Ok(HttpResponse::Ok().content_type("text/html").body(html_response))
}

/// Initializes and runs the Actix web server.
///
/// The server is configured with routes for the main page (`/`) and file uploads (`/upload`).
/// It binds to `127.0.0.1:8080` by default.
///
/// # Returns
/// A `std::io::Result<()>` which is `Ok(())` if the server runs successfully,
/// or an `Err` if there's an issue binding to the port or starting the server.
pub async fn run_server(port: u16) -> std::io::Result<()> {
    let server_address = format!("127.0.0.1:{}", port);
    println!("Starting server at http://{}/", server_address);
    HttpServer::new(|| {
        App::new()
            // Serve the main index.html page at the root
            .route("/", web::get().to(index))
            // Handle file uploads at /upload
            .route("/upload", web::post().to(upload_files))
    })
    .bind(server_address)?
}

// Helper function to process a single uploaded file
// Returns Ok((filename, info_map)) on successful parsing, where info_map includes a "_type" field.
// Returns Err(error_message_string) for parsing errors or if the file type is skipped.
async fn process_single_file(
    filename: &str, // Borrow filename
    file_bytes: Vec<u8>,
) -> Result<(String, HashMap<String, String>), String> { // Value of HashMap is String, so details will be stringified.
    if filename.ends_with(".safetensors") {
        // (Memory consideration comments remain applicable here)
        println!("Processing SafeTensors file: {}", filename);
        match SafeTensors::deserialize(&file_bytes) {
            Ok(tensors_metadata) => { // Renamed for clarity, this is metadata access
                let mut file_specific_info = HashMap::new();
                file_specific_info.insert("_type".to_string(), "safetensor".to_string());

                for (tensor_name, tensor_view) in tensors_metadata.tensors() {
                    let shape_str = format!("{:?}", tensor_view.shape());
                    let dtype_str = format!("{:?}", tensor_view.dtype()); // DType to String

                    // Preview generation
                    const MAX_PREVIEW_ELEMENTS: usize = 10;
                    let data_slice = tensor_view.data();
                    let mut preview_str = "[".to_string();

                    match tensor_view.dtype() {
                        safetensors::Dtype::F32 => {
                            let typed_data: &[f32] = bytemuck::cast_slice(data_slice);
                            for (i, val) in typed_data.iter().take(MAX_PREVIEW_ELEMENTS).enumerate() {
                                if i > 0 { preview_str.push_str(", "); }
                                preview_str.push_str(&format!("{:.4}", val));
                            }
                            if typed_data.len() > MAX_PREVIEW_ELEMENTS { preview_str.push_str(", ..."); }
                        }
                        safetensors::Dtype::I64 => {
                            let typed_data: &[i64] = bytemuck::cast_slice(data_slice);
                            for (i, val) in typed_data.iter().take(MAX_PREVIEW_ELEMENTS).enumerate() {
                                if i > 0 { preview_str.push_str(", "); }
                                preview_str.push_str(&val.to_string());
                            }
                            if typed_data.len() > MAX_PREVIEW_ELEMENTS { preview_str.push_str(", ..."); }
                        }
                        safetensors::Dtype::U8 => {
                            // Note: U8 can be raw bytes or actual uint8 data. Assuming numeric for preview.
                            for (i, val) in data_slice.iter().take(MAX_PREVIEW_ELEMENTS).enumerate() {
                                if i > 0 { preview_str.push_str(", "); }
                                preview_str.push_str(&val.to_string());
                            }
                            if data_slice.len() > MAX_PREVIEW_ELEMENTS { preview_str.push_str(", ..."); }
                        }
                        // Add other Dtypes as needed: F64, I32, U32, BF16, F16, etc.
                        // For unsupported DTypes for preview, or if data_slice is not aligned:
                        _ => {
                            preview_str.push_str("Preview not available for this dtype or data is misaligned");
                        }
                    }
                    preview_str.push(']');

                    file_specific_info.insert(format!("{}.shape", tensor_name), shape_str);
                    file_specific_info.insert(format!("{}.dtype", tensor_name), dtype_str);
                    file_specific_info.insert(format!("{}.preview", tensor_name), preview_str);
                }

                println!("Successfully processed SafeTensors file: {}", filename);
                Ok((filename.to_string(), file_specific_info))
            }
            Err(e) => {
                let error_msg = format!(
                    "<strong>File: '{}'</strong> - Invalid SafeTensors format. Details: {}. Please ensure the file is not corrupted and conforms to the SafeTensors specification.",
                    filename, e
                );
                eprintln!("Error deserializing SafeTensors file {}: {}", filename, e);
                Err(error_msg)
            }
        }
    } else if filename.ends_with("tokenizer.json") {
        // (Memory consideration comments remain applicable here)
        println!("Processing tokenizer.json file: {}", filename);
        match serde_json::from_slice::<Value>(&file_bytes) {
            Ok(json_value) => {
                let mut tokenizer_details = HashMap::new();
                tokenizer_details.insert("content_preview".to_string(), format!("{:.200}", json_value.to_string()));
                tokenizer_details.insert("_type".to_string(), "tokenizer".to_string());

                // Extract vocabulary preview
                if let Some(model) = json_value.get("model") {
                    if let Some(vocab_value) = model.get("vocab") {
                        if let Some(vocab_map) = vocab_value.as_object() {
                            // For vocab preview (first N tokens)
                            let mut vocab_preview_vec: Vec<(String, u32)> = vocab_map
                                .iter()
                                .filter_map(|(token, id_val)| {
                                    id_val.as_u64().map(|id| (token.clone(), id as u32))
                                })
                                .collect();
                            vocab_preview_vec.sort_by_key(|&(_, id)| id);
                            let vocab_preview_to_serialize = vocab_preview_vec
                                .into_iter()
                                .take(VOCAB_PREVIEW_COUNT)
                                .collect::<Vec<(String, u32)>>();
                            if let Ok(serialized_vocab_preview) = serde_json::to_string(&vocab_preview_to_serialize) {
                                tokenizer_details.insert("vocab_preview_json".to_string(), serialized_vocab_preview);
                            }

                            // For full vocabulary
                            let full_vocab_map: HashMap<String, u32> = vocab_map
                                .iter()
                                .filter_map(|(token, id_val)| {
                                    id_val.as_u64().map(|id| (token.clone(), id as u32))
                                })
                                .collect();
                            if let Ok(serialized_full_vocab) = serde_json::to_string(&full_vocab_map) {
                                tokenizer_details.insert("full_vocab_json".to_string(), serialized_full_vocab);
                            }
                        }
                    }
                }
                println!("Successfully processed tokenizer.json file: {}", filename);
                Ok((filename.to_string(), tokenizer_details))
            }
            Err(e) => {
                let error_msg = format!(
                    "<strong>File: '{}'</strong> - Invalid JSON format in tokenizer file. Details: {}. Please check for syntax errors like missing commas or brackets.",
                    filename, e
                );
                eprintln!("Error deserializing tokenizer.json file {}: {}", filename, e);
                Err(error_msg)
            }
        }
    } else {
        let skipped_msg = format!(
            "<strong>File: '{}'</strong> - Skipped. This file type is not supported. Please upload only .safetensors or tokenizer.json files.",
            filename
        );
        // This is treated as an "error" for reporting purposes to ensure the user is notified.
        Err(skipped_msg)
    }
}

/// Initializes and runs the Actix web server.
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web::Bytes, http::StatusCode};
    use actix_multipart::{Field, Multipart};
    use futures_util::stream::iter;
    use std::io::Write;
    use safetensors::serialize;
    use std::collections::HashMap;
    use actix_http::header::{CONTENT_TYPE, HeaderValue, HeaderName, DispositionType, ContentDisposition};


    // Helper function to create a multipart payload with a single file
    fn create_multipart_payload(filename: &str, content: Vec<u8>) -> Multipart {
        let field = Field::new(
            HeaderName::from_static("content-disposition"),
            HeaderValue::from_str(&format!("form-data; name=\"files\"; filename=\"{}\"", filename)).unwrap(),
            iter(vec![Ok(Bytes::from(content))])
        );
        Multipart::new(
            &HeaderValue::from_static("multipart/form-data; boundary=--boundary"),
            iter(vec![Ok(field)])
        )
    }

    // Helper function to create a multipart payload with multiple files
    fn create_multipart_payload_multiple(files: Vec<(&str, Vec<u8>)>) -> Multipart {
        let mut fields = Vec::new();
        for (filename, content) in files {
            let field = Field::new(
                HeaderName::from_static("content-disposition"),
                HeaderValue::from_str(&format!("form-data; name=\"files\"; filename=\"{}\"", filename)).unwrap(),
                iter(vec![Ok(Bytes::from(content))])
            );
            fields.push(Ok(field));
        }
        Multipart::new(
            &HeaderValue::from_static("multipart/form-data; boundary=--boundary"),
            iter(fields)
        )
    }


    #[actix_rt::test]
    async fn test_upload_tokenizer_json_success() {
        let tokenizer_json_content = r#"{
            "name": "test_tokenizer",
            "model": {
                "vocab": {
                    "<s>": 0,
                    "<pad>": 1,
                    "</s>": 2,
                    "a": 3,
                    "b": 4
                }
            }
        }"#;
        let payload = create_multipart_payload("tokenizer.json", tokenizer_json_content.as_bytes().to_vec());

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h3>tokenizer.json</h3>"));
        assert!(body_str.contains("test_tokenizer")); // Check for part of the content preview

        // Check for data-vocab-preview attribute on the <pre> tag
        let vocab_map_for_preview = vec![
            ("<s>".to_string(), 0u32),
            ("<pad>".to_string(), 1u32),
            ("</s>".to_string(), 2u32),
            ("a".to_string(), 3u32),
            ("b".to_string(), 4u32),
        ];
        let expected_vocab_preview_json_string = serde_json::to_string(&vocab_map_for_preview).unwrap();
        let expected_preview_attr_string = format!("data-vocab-preview='{}'", html_escape::encode_double_quoted_attribute(&expected_vocab_preview_json_string));
        assert!(body_str.contains(&expected_preview_attr_string), "Expected vocab preview attribute not found or incorrect. Body: {}", body_str);

        // Check for data-full-vocab-json attribute
        let full_vocab_map_for_assert: HashMap<String, u32> = vocab_map_for_preview.into_iter().collect();
        let expected_full_vocab_json_string = serde_json::to_string(&full_vocab_map_for_assert).unwrap();
        let expected_full_attr_string = format!("data-full-vocab-json='{}'", html_escape::encode_double_quoted_attribute(&expected_full_vocab_json_string));
        assert!(body_str.contains(&expected_full_attr_string), "Expected full vocab attribute not found or incorrect. Body: {}", body_str);
    }

    #[actix_rt::test]
    async fn test_upload_tokenizer_json_success_empty_vocab() {
        let tokenizer_json_content = r#"{"name": "test_tokenizer_no_vocab", "model": {}}"#; // No "vocab" field
        let payload = create_multipart_payload("tokenizer_no_vocab.json", tokenizer_json_content.as_bytes().to_vec());

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h3>tokenizer_no_vocab.json</h3>"));
        assert!(!body_str.contains("data-vocab-preview="), "data-vocab-preview should not be present. Body: {}", body_str);
        assert!(!body_str.contains("data-full-vocab-json="), "data-full-vocab-json should not be present. Body: {}", body_str);
    }

    #[actix_rt::test]
    async fn test_upload_tokenizer_json_success_many_vocab_truncated() {
        let mut vocab_map_str = String::from("{");
        for i in 0..(VOCAB_PREVIEW_COUNT + 5) {
            vocab_map_str.push_str(&format!("\"token{}\": {},", i, i));
        }
        vocab_map_str.pop(); // Remove last comma
        vocab_map_str.push('}');
        let tokenizer_json_content = format!(r#"{{
            "name": "test_tokenizer_many_vocab",
            "model": {{ "vocab": {} }}
        }}"#, vocab_map_str);

        let payload = create_multipart_payload("tokenizer_many_vocab.json", tokenizer_json_content.as_bytes().to_vec());
        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h3>tokenizer_many_vocab.json</h3>"));

        let data_attr_start = "data-vocab-preview='";
        let data_attr_end = "']"; // End of the JSON array within the attribute
        if let Some(start_idx) = body_str.find(data_attr_start) {
            let actual_attr_val_with_end = &body_str[start_idx + data_attr_start.len()..];
            if let Some(end_idx) = actual_attr_val_with_end.find(data_attr_end) { // Find closing bracket of the array value
                 let actual_json_str = &actual_attr_val_with_end[..end_idx + 1];
                 let parsed_vocab_preview: Vec<(String, u32)> = serde_json::from_str(actual_json_str).expect("Failed to parse vocab preview from HTML attribute");
                 assert_eq!(parsed_vocab_preview.len(), VOCAB_PREVIEW_COUNT, "Vocabulary preview should be truncated to VOCAB_PREVIEW_COUNT");
                 for i in 0..VOCAB_PREVIEW_COUNT {
                     assert_eq!(parsed_vocab_preview[i].0, format!("token{}", i));
                     assert_eq!(parsed_vocab_preview[i].1, i as u32);
                 }
            } else {
                panic!("Could not find end of data-vocab-preview attribute value. Body: {}", body_str);
            }
        } else {
            panic!("data-vocab-preview attribute not found. Body: {}", body_str);
        }

        // Check full vocab
        let full_data_attr_start = "data-full-vocab-json='{"; // Starts with object
        let full_data_attr_end = "}'"; // End of the JSON object within the attribute
         if let Some(start_idx) = body_str.find(full_data_attr_start) {
            let actual_attr_val_with_end = &body_str[start_idx + full_data_attr_start.len() -1 ..]; // include the starting '{'
            if let Some(end_idx) = actual_attr_val_with_end.find(full_data_attr_end) {
                 let actual_json_str = &actual_attr_val_with_end[..end_idx + 1]; // Include the closing '}'
                 let parsed_full_vocab: HashMap<String, u32> = serde_json::from_str(actual_json_str).expect("Failed to parse full vocab from HTML attribute");
                 assert_eq!(parsed_full_vocab.len(), VOCAB_PREVIEW_COUNT + 5, "Full vocabulary should contain all items");
                 for i in 0..(VOCAB_PREVIEW_COUNT + 5) {
                     let token_name = format!("token{}", i);
                     assert_eq!(parsed_full_vocab.get(&token_name), Some(&(i as u32)));
                 }
            } else {
                panic!("Could not find end of data-full-vocab-json attribute value. Body: {}", body_str);
            }
        } else {
            panic!("data-full-vocab-json attribute not found. Body: {}", body_str);
        }
    }


    #[actix_rt::test]
    async fn test_upload_safetensors_success() {
        // Create a safetensors file with one tensor
        let mut tensors_map = HashMap::new();
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![2, 2],
            bytemuck::bytes_of_slice(&tensor_data),
        ).unwrap();
        tensors_map.insert("weight1".to_string(), tensor_view);

        let safetensor_bytes = serialize(&tensors_map, &None)
            .expect("Failed to serialize tensor for test");

        let payload = create_multipart_payload("test.safetensors", safetensor_bytes);
        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h3>test.safetensors</h3>"));
        assert!(body_str.contains("<ul>"));
        let tensor_id_val = "test.safetensors::weight1";
        let expected_li = format!(
            "<li data-dtype=\"F32\" data-preview=\"[1.0000, 2.0000, 3.0000, 4.0000]\" data-shape=\"[2, 2]\" data-tensor-id=\"{}\">"
            + "<input type=\"checkbox\" name=\"tensor_compare_select\" value=\"{}\"> "
            + "weight1: [2, 2]</li>",
            html_escape::encode_double_quoted_attribute(tensor_id_val),
            html_escape::encode_double_quoted_attribute(tensor_id_val)
        );
        assert!(body_str.contains(&expected_li), "Generated HTML: {}\nExpected LI: {}", body_str, expected_li);
        assert!(body_str.contains("</ul>"));
    }

    #[actix_rt::test]
    async fn test_upload_tokenizer_json_corrupted() {
        let corrupted_content = r#"{"name": "test_tokenizer",}"#.as_bytes().to_vec(); // Invalid JSON
        let payload = create_multipart_payload("tokenizer_corrupted.json", corrupted_content);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'tokenizer_corrupted.json'</strong> - Invalid JSON format in tokenizer file."));
    }

    #[actix_rt::test]
    async fn test_upload_safetensors_corrupted() {
        let corrupted_bytes = b"this is not a valid safetensors file".to_vec();
        let payload = create_multipart_payload("corrupted.safetensors", corrupted_bytes);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'corrupted.safetensors'</strong> - Invalid SafeTensors format."));
    }

    #[actix_rt::test]
    async fn test_upload_unknown_file_type() {
        let other_content = "This is a plain text file.".as_bytes().to_vec();
        let payload = create_multipart_payload("other.txt", other_content);

        let resp = upload_files(payload).await.unwrap();
        // Now expects BAD_REQUEST because skipped files are treated as processing issues.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'other.txt'</strong> - Skipped. This file type is not supported."));
        // Check that there's no "Successfully Processed Files" section if only a skip occurs.
        assert!(!body_str.contains("<hr><h3>Successfully Processed Files (if any):</h3>"));
    }

    #[actix_rt::test]
    async fn test_upload_mixed_files_valid_and_invalid() {
        let valid_tokenizer_content = r#"{"name": "valid_tokenizer"}"#.as_bytes().to_vec();
        let corrupted_safetensor_bytes = b"invalid safetensor data".to_vec();
        let other_content = "plain text".as_bytes().to_vec();

        let files = vec![
            ("tokenizer.json", valid_tokenizer_content),
            ("bad.safetensors", corrupted_safetensor_bytes),
            ("notes.txt", other_content),
        ];
        let payload = create_multipart_payload_multiple(files);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST); // Because one file is bad

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'bad.safetensors'</strong> - Invalid SafeTensors format."));
        assert!(body_str.contains("<strong>File: 'notes.txt'</strong> - Skipped. This file type is not supported."));

        assert!(body_str.contains("<hr><h3>Successfully Processed Files (if any):</h3>"));
        assert!(body_str.contains("<h3>tokenizer.json</h3>"));
        assert!(body_str.contains(r#"name&quot;: &quot;valid_tokenizer&quot;"#));

        // Ensure notes.txt was skipped and not listed as processed or error
        assert!(!body_str.contains("<h3>notes.txt</h3>"));
    }
     #[actix_rt::test]
    async fn test_upload_multiple_valid_files() {
        let tokenizer_content = r#"{"name": "multi_tokenizer"}"#.as_bytes().to_vec();

        let mut sf_tensors_map = HashMap::new();
        let sf_tensor_data: Vec<i64> = vec![100, 200, 300];
        let sf_tensor_view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::I64,
            vec![3],
            bytemuck::bytes_of_slice(&sf_tensor_data),
        ).unwrap();
        sf_tensors_map.insert("data_tensor".to_string(), sf_tensor_view);
        let safetensor_bytes = serialize(&sf_tensors_map, &None)
            .expect("Failed to serialize tensor for multi-file test");

        let files = vec![
            ("good_tokenizer.json", tokenizer_content),
            ("good.safetensors", safetensor_bytes),
        ];
        let payload = create_multipart_payload_multiple(files);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h2>Uploaded File Information:</h2>"));
        // Check tokenizer file part
        assert!(body_str.contains("<h3>good_tokenizer.json</h3>"));
        assert!(body_str.contains(r#"name&quot;: &quot;multi_tokenizer&quot;"#));

        // Check safetensor file part
        assert!(body_str.contains("<h3>good.safetensors</h3>"));
        let sf_tensor_id_val = "good.safetensors::data_tensor";
        let expected_sf_li = format!(
            "<li data-dtype=\"I64\" data-preview=\"[100, 200, 300]\" data-shape=\"[3]\" data-tensor-id=\"{}\">"
            + "<input type=\"checkbox\" name=\"tensor_compare_select\" value=\"{}\"> "
            + "data_tensor: [3]</li>",
            html_escape::encode_double_quoted_attribute(sf_tensor_id_val),
            html_escape::encode_double_quoted_attribute(sf_tensor_id_val)
        );
        assert!(body_str.contains(&expected_sf_li), "Generated HTML for safetensor in multi-upload: {}\nExpected LI: {}", body_str, expected_sf_li);
    }
}
