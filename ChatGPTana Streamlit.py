import json
import streamlit as st
from datetime import datetime
from io import BytesIO
import zipfile

def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('[[date:%Y-%m-%d %H:%M]]')

def format_nested_content(content, indent_level=4):
    lines = content.split('\n')
    formatted_content = ""
    for line in lines:
        if line.strip():  # Ignore empty lines
            formatted_content += " " * indent_level + line.strip() + "\n"
    return formatted_content

def json_to_tana_paste(json_data):
    tana_paste = "%%tana%%\n\n"
    
    for conversation in json_data:
        title = conversation.get("title", "No Title")
        create_time = conversation.get("create_time", None)
        update_time = conversation.get("update_time", None)
        mapping = conversation.get("mapping", {})
        
        tana_paste += f"- {title} #chatgpt\n"
        tana_paste += f"  - Title:: {title}\n"
        if create_time:
            tana_paste += f"  - Created Time:: {convert_timestamp(create_time)}\n"
        if update_time:
            tana_paste += f"  - Updated Time:: {convert_timestamp(update_time)}\n"
        
        tana_paste += f"  - Chat::\n"
        
        for node_id, node in mapping.items():
            message = node.get("message", None)
            if message is None:
                continue
            
            author_role = message.get("author", {}).get("role", "")
            content_parts = message.get("content", {}).get("parts", [])
            
            if author_role == "user":
                author_prefix = "Self"
            else:
                author_prefix = "ChatGPT"
            
            for part in content_parts:
                if isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
                    image_url = part.get("asset_pointer", "")
                    if image_url:
                        tana_paste += f"    - {author_prefix}: ![]({image_url})\n"
                elif isinstance(part, str):
                    content_text = part if part else ""
                    if content_text.strip():
                        # Check for nested content and format it
                        if "\n" in content_text:
                            formatted_content = format_nested_content(content_text, indent_level=6)
                            tana_paste += f"    - {author_prefix}:\n{formatted_content}\n"
                        else:
                            tana_paste += f"    - {author_prefix}: {content_text}\n"
        
        tana_paste += "\n"
    
    return tana_paste

def split_tana_paste(tana_paste, max_chars=100000):
    conversations = tana_paste.split("\n- ")  # Split by conversation
    files = []
    current_file = "%%tana%%\n\n"
    current_length = len(current_file)

    for conversation in conversations:
        conversation = conversation.strip()
        if not conversation:
            continue
        
        conversation_length = len(conversation) + len("\n- ")
        
        if current_length + conversation_length > max_chars:
            files.append(current_file)
            current_file = "%%tana%%\n\n"
            current_length = len(current_file)
        
        current_file += "- " + conversation + "\n\n"
        current_length += conversation_length
    
    if current_file.strip():
        files.append(current_file)
    
    return files

st.title("JSON to Tana Paste Converter")
st.write("Upload a JSON file and convert it to Tana Paste format. The converted file will be split into multiple files if it exceeds 100k characters.")

uploaded_file = st.file_uploader("Choose a JSON file", type="json")

if uploaded_file is not None:
    json_data = json.load(uploaded_file)
    
    if st.button("Convert"):
        tana_paste = json_to_tana_paste(json_data)
        files = split_tana_paste(tana_paste)
        
        # Create a ZIP file to store all split files
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, file_content in enumerate(files):
                zip_file.writestr(f"converted_tana_paste_part_{i+1}.txt", file_content)
        
        st.write("Conversion Successful! Download the Tana Paste files below:")
        
        st.download_button(
            label="Download Tana Paste ZIP",
            data=zip_buffer.getvalue(),
            file_name="converted_tana_paste.zip",
            mime="application/zip"
        )