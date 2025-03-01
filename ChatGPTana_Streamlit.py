import json
import streamlit as st
from datetime import datetime
from io import BytesIO
import zipfile
import re

################################################################################
# Configuration
################################################################################

MAX_CHARS_PER_FILE = 80000    # Maximum characters per Tana Paste file chunk

################################################################################
# Helper Functions
################################################################################

def convert_timestamp(ts):
    """Convert a Unix timestamp to Tana's date format."""
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime('[[date:%Y-%m-%d %H:%M]]')

def remove_extra_tana_tags(text):
    """Remove occurrences of '%%tana%%' from text."""
    return text.replace("%%tana%%", "")

def format_multiline_for_tana(text, indent_level=6):
    """
    Indent multiline text so Tana interprets it as nested lines.
    Triple-backtick code blocks are preserved.
    """
    lines = text.split("\n")
    return "\n".join(" " * indent_level + line.strip() for line in lines if line.strip())

def extract_text(msg):
    """
    Extract and combine text from a message's 'parts'. 
    If a part is a dict and has a "text" key (e.g. for audio transcriptions), that text is extracted.
    Otherwise, the part is converted to a string.
    """
    parts = msg.get("content", {}).get("parts", [])
    extracted = []
    for part in parts:
        if isinstance(part, dict):
            if "text" in part:
                extracted.append(part["text"])
            else:
                extracted.append(str(part))
        else:
            extracted.append(str(part))
    return remove_extra_tana_tags(" ".join(extracted)).strip()

################################################################################
# Merged JSON -> Tana Paste Conversion (Fast Merging)
################################################################################

def json_to_tana_paste_merged_fast(json_data):
    out_lines = []
    out_lines.append("%%tana%%")
    out_lines.append("")  # blank line

    for conv in json_data:
        title = conv.get("title", "No Title")
        ctime = conv.get("create_time")
        utime = conv.get("update_time")
        mapping = conv.get("mapping", {})

        # Build conversation header
        conv_header = []
        conv_header.append(f"- {title} #chatgpt")
        conv_header.append(f"  - Title:: {title}")
        if ctime:
            conv_header.append(f"  - Created Time:: {convert_timestamp(ctime)}")
        if utime:
            conv_header.append(f"  - Updated Time:: {convert_timestamp(utime)}")
        conv_header.append("  - Chat::")
        conv_header_text = "\n".join(conv_header)
        out_lines.append(conv_header_text)
        out_lines.append("")  # blank line

        # Gather and sort messages
        messages = []
        for node in mapping.values():
            msg = node.get("message")
            if msg:
                messages.append(msg)
        messages.sort(key=lambda m: m.get("create_time") or 0)

        pending_user = None
        assistant_buffer = []  # accumulate all assistant/tool texts for current Q–A block

        def finalize_qa():
            nonlocal pending_user, assistant_buffer
            if pending_user and assistant_buffer:
                user_text = extract_text(pending_user)
                # Check if user text is wrapped in quotes => voice
                was_quoted = ((user_text.startswith("“") and user_text.endswith("”")) or
                              (user_text.startswith('"') and user_text.endswith('"')))
                if was_quoted:
                    user_text = user_text[1:-1].strip()
                chat_type = "voice" if was_quoted else "text"
                q_time = pending_user.get("create_time")
                q_time_formatted = convert_timestamp(q_time) if q_time else ""
                merged_answer = "\n".join(assistant_buffer).strip()

                out_lines.append("    - #chatgptquestion")
                out_lines.append("      - question::")
                for line in format_multiline_for_tana(user_text, indent_level=8).splitlines():
                    out_lines.append(line)
                if q_time_formatted:
                    if chat_type == "voice":
                        out_lines.append(f"      - audio_length:: {q_time_formatted}")
                    else:
                        out_lines.append(f"      - question_time:: {q_time_formatted}")
                out_lines.append(f"      - chat_type:: {chat_type}")
                out_lines.append("      - answer::")
                for line in format_multiline_for_tana(merged_answer, indent_level=8).splitlines():
                    out_lines.append(line)
                out_lines.append("")  # blank line for separation

                pending_user = None
                assistant_buffer = []

        for msg in messages:
            role = msg.get("author", {}).get("role", "")
            text = extract_text(msg)
            if not text:
                continue

            if role == "user":
                finalize_qa()  # finalize previous pairing if exists
                pending_user = msg
                assistant_buffer = []
            elif role in ("assistant", "tool"):
                if pending_user:
                    assistant_buffer.append(text)
                else:
                    out_lines.append("    - ChatGPT::")
                    for line in format_multiline_for_tana(text, indent_level=8).splitlines():
                        out_lines.append(line)
            else:
                continue

        finalize_qa()  # finalize any leftover pairing for this conversation
        out_lines.append("")  # blank line between conversations

    return "\n".join(out_lines)

################################################################################
# Splitting by Complete Conversation Nodes
################################################################################

def split_conversation_by_qna(conv_text, max_chars):
    """
    Split a single conversation node into chunks by complete Q–A block boundaries.
    The conversation node is assumed to have a header (everything up to and including "  - Chat::")
    and then Q–A blocks starting with "    - #chatgptquestion".
    Each chunk will re‑include the header.
    """
    lines = conv_text.splitlines()
    header_lines = []
    qna_blocks = []
    current_qna = []
    header_found = False
    for line in lines:
        if not header_found:
            header_lines.append(line)
            if "  - Chat::" in line:
                header_found = True
        else:
            if line.strip().startswith("- #chatgptquestion") or line.strip().startswith("    - #chatgptquestion"):
                if current_qna:
                    qna_blocks.append("\n".join(current_qna))
                    current_qna = []
            current_qna.append(line)
    if current_qna:
        qna_blocks.append("\n".join(current_qna))
    header_text = "\n".join(header_lines)
    
    chunks = []
    current_chunk = []
    current_length = len(header_text) + 2  # header plus two newlines
    def finalize_chunk(chunk_lines):
        return header_text + "\n\n" + "\n\n".join(chunk_lines)
    
    for block in qna_blocks:
        block_len = len(block) + 2  # plus separating newlines
        if current_length + block_len > max_chars and current_chunk:
            chunks.append(finalize_chunk(current_chunk))
            current_chunk = []
            current_length = len(header_text) + 2
        current_chunk.append(block)
        current_length += block_len
    if current_chunk:
        chunks.append(finalize_chunk(current_chunk))
    return chunks

def split_tana_by_conversations(tana_text, max_chars=MAX_CHARS_PER_FILE):
    """
    Split the full Tana Paste output into chunks by complete conversation nodes.
    If a conversation node exceeds max_chars, split it further by complete Q–A blocks
    (using split_conversation_by_qna), ensuring that each chunk ends with a complete #chatgptquestion block.
    Each final chunk is prefixed with "%%tana%%" on its own line.
    """
    overall_header = "%%tana%%\n\n"
    if tana_text.startswith(overall_header):
        body = tana_text[len(overall_header):]
    else:
        body = tana_text

    # Split by conversation nodes (each conversation node starts with a line beginning with "- ")
    convos = body.split("\n\n- ")
    if convos:
        convos = [convos[0]] + ["- " + convo for convo in convos[1:]]
    else:
        convos = []

    chunks = []
    current_chunk = []
    current_length = 0

    def finalize_chunk(chunk_list):
        return overall_header + "\n\n".join(chunk_list)

    for convo in convos:
        if len(convo) > max_chars:
            subchunks = split_conversation_by_qna(convo, max_chars)
            for sub in subchunks:
                extra = 2 if current_chunk else 0
                if current_length + len(sub) + extra > max_chars and current_chunk:
                    chunks.append(finalize_chunk(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sub)
                current_length += len(sub) + extra
            continue
        extra = 2 if current_chunk else 0
        if current_length + len(convo) + extra > max_chars and current_chunk:
            chunks.append(finalize_chunk(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(convo)
        current_length += len(convo) + extra

    if current_chunk:
        chunks.append(finalize_chunk(current_chunk))
    return chunks

################################################################################
# Fallback Line-by-Line Splitter (if needed)
################################################################################

def split_tana_paste(tana_text, max_chars=MAX_CHARS_PER_FILE):
    lines = tana_text.splitlines()
    if lines and lines[0].strip() == "%%tana%%":
        lines = lines[1:]
    chunks = []
    current_chunk = []
    current_length = 0
    def finalize(chunk_lines):
        return "%%tana%%\n\n" + "\n".join(chunk_lines).strip()
    for line in lines:
        line_len = len(line) + 1
        if current_length + line_len > max_chars and current_chunk:
            chunks.append(finalize(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_len
    if current_chunk:
        chunks.append(finalize(current_chunk))
    return chunks

################################################################################
# Manual Chat Conversion (Unchanged)
################################################################################

def manual_chat_to_tana_paste(manual_text):
    tana = "%%tana%%\n\n"
    blocks = [block.strip() for block in manual_text.split("\n\n") if block.strip()]
    for block in blocks:
        if "ChatGPT said:" not in block:
            continue
        lines = block.splitlines()
        try:
            answer_index = next(i for i, line in enumerate(lines) if "ChatGPT said:" in line)
        except StopIteration:
            continue
        question_lines = lines[:answer_index]
        question_time = None
        processed_q = []
        for line in question_lines:
            line = line.strip()
            if "You said:" in line:
                line = line.replace("You said:", "").strip()
            if re.match(r'^\d{1,2}:\d{2}$', line):
                if not question_time:
                    question_time = line
                continue
            processed_q.append(line)
        question_text = "\n".join(processed_q).strip()
        answer_lines = lines[answer_index+1:]
        processed_a = []
        answer_time = None
        for line in answer_lines:
            line = line.strip()
            if re.match(r'^\d{1,2}:\d{2}$', line):
                if not answer_time:
                    answer_time = line
                continue
            processed_a.append(line)
        answer_text = "\n".join(processed_a).strip()
        tana += f"- #chatgptquestion\n"
        tana += "  - question::\n" + format_multiline_for_tana(question_text, indent_level=6) + "\n"
        if question_time:
            tana += f"  - question_time:: {question_time}\n"
        tana += "  - chat_type:: text\n"
        tana += "  - answer::\n" + format_multiline_for_tana(answer_text, indent_level=6) + "\n\n"
    return tana

################################################################################
# Streamlit App
################################################################################

st.title("ChatGPT to Tana Paste Converter")
st.write(
    "Upload your JSON file and convert it into Tana Paste.\n\n"
    "This version merges all assistant/tool messages between user messages into one answer,\n"
    "and splits the output by complete conversation nodes so that each file starts with the full conversation header\n"
    "and ends with a complete #chatgptquestion block. If a conversation node is too long, it will be split by Q–A blocks,\n"
    "with its header repeated in subsequent chunks."
)

conversion_mode = st.radio("Select conversion mode:", ["JSON File", "Manual Chat"])

if conversion_mode == "JSON File":
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    raw_json_str = ""
    if uploaded_file is not None:
        raw_json_bytes = uploaded_file.read()
        raw_json_str = raw_json_bytes.decode("utf-8", errors="replace")
        # JSON preview removed per request.
        json_data = json.loads(raw_json_str)
    
    if st.button("Convert JSON"):
        if not raw_json_str.strip():
            st.write("No JSON loaded or file is empty.")
        else:
            tana_text = json_to_tana_paste_merged_fast(json_data)
            files = split_tana_by_conversations(tana_text, max_chars=MAX_CHARS_PER_FILE)
            if files:
                st.subheader("Preview of First Tana Paste File (After Conversion)")
                st.text_area("Copy/Paste Preview (File 1)", value=files[0], height=400)
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for i, content in enumerate(files):
                        zipf.writestr(f"converted_tana_paste_part_{i+1}.txt", content)
                st.write("Conversion Successful! Download the Tana Paste files below:")
                st.download_button(
                    label="Download Tana Paste ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="converted_tana_paste.zip",
                    mime="application/zip"
                )
            else:
                st.write("No content found or empty after parsing.")
elif conversion_mode == "Manual Chat":
    manual_text = st.text_area("Paste your chat transcript here", height=400,
                               help="Each Q–A pair is processed, with multiline support.")
    if st.button("Convert Manual Chat"):
        if not manual_text.strip():
            st.write("Please paste some chat text first.")
        else:
            tana_text = manual_chat_to_tana_paste(manual_text)
            st.subheader("Tana Paste Output")
            st.text_area("Copy the Tana Paste below:", value=tana_text, height=400)
