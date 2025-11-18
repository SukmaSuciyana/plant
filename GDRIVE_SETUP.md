# Google Drive Setup Instructions

## Overview
The application is configured to automatically download model files from Google Drive if they don't exist locally. This is useful for:
- Streamlit Cloud deployment (where large files can't be stored in Git)
- Fresh installations without downloading the full repository

## Steps to Configure Google Drive File IDs

### 1. Upload Model Files to Google Drive
Upload the following files to your Google Drive and make them publicly accessible:
- `saved_model_format/saved_model.pb`
- `saved_model_format/variables/variables.index`
- `saved_model_format/variables/variables.data-00000-of-00001`
- `saved_model_format/fingerprint.pb`
- (Optional) `model/model_info.json`

### 2. Make Files Public
For each file:
1. Right-click the file in Google Drive
2. Select "Share"
3. Click "Change to anyone with the link"
4. Set permission to "Viewer"
5. Copy the link

### 3. Extract File IDs
From a Google Drive link like:
```
https://drive.google.com/file/d/1ABC123xyz-EXAMPLE-FILE-ID/view?usp=sharing
```

The file ID is: `1ABC123xyz-EXAMPLE-FILE-ID`

### 4. Update app.py
Open `main/app.py` and find the `MODEL_FILES` dictionary in the `ensure_model_files()` function:

```python
MODEL_FILES = {
    "saved_model_format/saved_model.pb": "<GDRIVE_FILE_ID_SAVED_MODEL_PB>",
    "saved_model_format/variables/variables.index": "<GDRIVE_FILE_ID_VARIABLES_INDEX>",
    "saved_model_format/variables/variables.data-00000-of-00001": "<GDRIVE_FILE_ID_VARIABLES_DATA>",
    "saved_model_format/fingerprint.pb": "<GDRIVE_FILE_ID_FINGERPRINT_PB>",
}
```

Replace the placeholder IDs with your actual Google Drive file IDs:

```python
MODEL_FILES = {
    "saved_model_format/saved_model.pb": "1ABC123xyz-YOUR-ACTUAL-ID",
    "saved_model_format/variables/variables.index": "1DEF456abc-YOUR-ACTUAL-ID",
    "saved_model_format/variables/variables.data-00000-of-00001": "1GHI789def-YOUR-ACTUAL-ID",
    "saved_model_format/fingerprint.pb": "1JKL012ghi-YOUR-ACTUAL-ID",
}
```

### 5. Test Locally
Before deployment, test locally:
1. Delete your local model files (or move them temporarily)
2. Run the app: `streamlit run main/app.py`
3. The app should automatically download missing files

## Alternative: Using .gitignore
If you don't want to track large model files in Git:
1. Add to `.gitignore`:
   ```
   saved_model_format/variables/variables.data-00000-of-00001
   saved_model_format/saved_model.pb
   ```
2. Keep other small files in Git (fingerprint.pb, variables.index, model_info.json)
3. Only configure Google Drive IDs for the large files

## Troubleshooting

### Download Fails
- Ensure files are set to "Anyone with the link can view"
- Check file IDs are correct
- Verify you have `gdown` installed: `pip install gdown`

### Files Not Found After Download
- Check the paths in `MODEL_FILES` dictionary match your actual folder structure
- Verify download completed successfully (check file sizes)

### Rate Limiting
If downloading many times during testing, Google Drive may rate limit. Wait a few minutes or use a different account.

## Notes
- The `ensure_model_files()` function only downloads files that don't exist locally
- Files are checked and downloaded when `load_model()` is called
- The download happens inside the `@st.cache_resource` decorator, so it only runs once per session
