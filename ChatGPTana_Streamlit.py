import json
import streamlit as st
from datetime import datetime
from io import BytesIO
import zipfile
import re
import time

################################################################################
# Configuration
################################################################################

MAX_FILE_SIZE = 100000    # Maximum allowed characters per TXT file
MIN_FILE_SIZE = 50000     # Aim for no more than one file under this if mergeable

################################################################################
# Helper Functions
################################################################################

def convert_timestamp(ts):
    """Convert a Unix timestamp to Tana's date format."""
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime('[[date:%Y-%m-%d %H:%M]]')

def remove_extra_tana_tags(text):
    """Remove any '%%tana%%' placeholders from the text."""
    return text.replace("%%tana%%", "")

def format_multiline_as_nodes(text, indent_level=8):
    """
    Split the given text into individual lines and output each as its own bullet node.
    Each non-empty line becomes a node prefixed with "- ".
    """
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(" " * indent_level + "- " + line.strip() for line in lines if line.strip())

def extract_text(msg):
    """
    Extract and combine text from a message's 'parts'.
    If a part is a dict and has a "text" key (e.g. for audio transcriptions),
    that text is used; otherwise, the part is stringified.
    """
    try:
        parts = msg.get("content", {}).get("parts", [])
        extracted = []
        for part in parts:
            if isinstance(part, dict):
                extracted.append(part.get("text", str(part)))
            else:
                extracted.append(str(part))
        return remove_extra_tana_tags(" ".join(extracted)).strip()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def filter_canvas_json(tana_text):
    """
    Process the Tana Paste text and replace canvas JSON blocks.
    If a line parses as JSON and has "type": "document" with a "content" key,
    it is replaced with that content.
    """
    filtered_lines = []
    for line in tana_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if obj.get("type") == "document" and "content" in obj:
                    indent = line[:line.index("{")] if "{" in line else ""
                    filtered_lines.append(indent + obj["content"])
                    continue
            except Exception:
                pass
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

def force_split_large_qna_block(qa_block, available):
    """
    Force-split a Q–A block (as a string) into sub-blocks that each fit within the available character count.
    Splitting is done by line boundaries.
    """
    lines = qa_block.splitlines()
    subblocks = []
    current_block = []
    current_length = 0
    for line in lines:
        line_length = len(line) + 1  # +1 for newline
        if current_length + line_length > available and current_block:
            subblocks.append("\n".join(current_block))
            current_block = []
            current_length = 0
        current_block.append(line)
        current_length += line_length
    if current_block:
        subblocks.append("\n".join(current_block))
    return subblocks

################################################################################
# Conversion: Merge Q–A Blocks
################################################################################

def json_to_tana_paste_merged_fast(json_data, progress_bar=None):
    """
    Merge consecutive assistant/tool messages following a user message into one Q–A block.
    Every line in both question and answer fields becomes its own bullet node.
    """
    out_lines = ["%%tana%%", ""]  # Start with Tana header

    total_convs = len(json_data)
    for i, conv in enumerate(json_data):
        if progress_bar:
            progress_bar.progress((i+1)/total_convs)
        title = conv.get("title", "No Title")
        ctime = conv.get("create_time")
        utime = conv.get("update_time")
        mapping = conv.get("mapping", {})

        # Build conversation header
        conv_header = [
            f"- {title} #chatgpt",
            f"  - Title:: {title}"
        ]
        if ctime:
            conv_header.append(f"  - Created Time:: {convert_timestamp(ctime)}")
        if utime:
            conv_header.append(f"  - Updated Time:: {convert_timestamp(utime)}")
        conv_header.append("  - Chat::")
        conv_header_text = "\n".join(conv_header)
        out_lines.append(conv_header_text)
        out_lines.append("")

        # Gather and sort messages
        messages = []
        for node in mapping.values():
            msg = node.get("message")
            if msg:
                messages.append(msg)
        messages.sort(key=lambda m: m.get("create_time") or 0)

        pending_user = None
        assistant_buffer = []

        def finalize_qa():
            nonlocal pending_user, assistant_buffer
            if pending_user and assistant_buffer:
                user_text = extract_text(pending_user)
                quoted = (user_text.startswith(("“", '"')) and user_text.endswith(("”", '"')))
                if quoted:
                    user_text = user_text[1:-1].strip()
                chat_type = "voice" if quoted else "text"
                q_time = pending_user.get("create_time")
                q_time_str = convert_timestamp(q_time) if q_time else ""
                merged_answer = "\n".join(assistant_buffer).strip()

                qa_lines = []
                qa_lines.append("    - #chatgptquestion")
                qa_lines.append("      - question::")
                qa_lines.append(format_multiline_as_nodes(user_text, indent_level=8))
                if q_time_str:
                    if chat_type == "voice":
                        qa_lines.append(f"      - audio_length:: {q_time_str}")
                    else:
                        qa_lines.append(f"      - question_time:: {q_time_str}")
                qa_lines.append(f"      - chat_type:: {chat_type}")
                qa_lines.append("      - answer::")
                qa_lines.append(format_multiline_as_nodes(merged_answer, indent_level=8))
                qa_block = "\n".join(qa_lines)
                
                # Force-split if necessary
                header_len = len("\n".join(conv_header)) + 2
                if len(qa_block) + header_len > MAX_FILE_SIZE:
                    available = MAX_FILE_SIZE - header_len - 2
                    subblocks = force_split_large_qna_block(qa_block, available)
                    for sb in subblocks:
                        out_lines.append(sb)
                        out_lines.append("")
                else:
                    out_lines.append(qa_block)
                    out_lines.append("")
                pending_user = None
                assistant_buffer.clear()

        for msg in messages:
            try:
                role = msg.get("author", {}).get("role", "")
                text = extract_text(msg)
                if not text:
                    continue
                if role == "user":
                    finalize_qa()
                    pending_user = msg
                    assistant_buffer.clear()
                elif role in ("assistant", "tool"):
                    if pending_user:
                        assistant_buffer.append(text)
                    else:
                        out_lines.append("    - ChatGPT::")
                        out_lines.append(format_multiline_as_nodes(text, indent_level=8))
                else:
                    continue
            except Exception:
                continue
        finalize_qa()
        out_lines.append("")

    merged_text = "\n".join(out_lines)
    return filter_canvas_json(merged_text)

################################################################################
# Splitting: Conversation Node Level
################################################################################

def split_conversation_by_qna(conv_text, max_chars):
    """
    Split a single conversation node into chunks by complete Q–A block boundaries.
    The conversation node is assumed to have a header (everything up to and including the line with "  - Chat::")
    and then Q–A blocks that start with a line containing "#chatgptquestion".
    Each chunk will re-include the header.
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
            if (line.strip().startswith("- #chatgptquestion") or 
                line.strip().startswith("    - #chatgptquestion")):
                if current_qna:
                    block = "\n".join(current_qna)
                    if len(block) > max_chars:
                        subblocks = force_split_large_qna_block(block, max_chars)
                        qna_blocks.extend(subblocks)
                    else:
                        qna_blocks.append(block)
                    current_qna = []
            current_qna.append(line)
    if current_qna:
        block = "\n".join(current_qna)
        if len(block) > max_chars:
            subblocks = force_split_large_qna_block(block, max_chars)
            qna_blocks.extend(subblocks)
        else:
            qna_blocks.append(block)
    
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

def split_tana_by_conversations(tana_text, max_chars):
    """
    Split the full Tana Paste output into chunks by conversation boundaries.
    If a conversation node exceeds max_chars, split it further by Q–A boundaries
    so that each chunk ends with a complete #chatgptquestion block.
    Each final chunk is prefixed with "%%tana%%" on its own line.
    """
    overall_header = "%%tana%%\n\n"
    body = tana_text[len(overall_header):] if tana_text.startswith(overall_header) else tana_text
    convos = body.split("\n\n- ")
    if convos:
        convos = [convos[0]] + ["- " + convo for convo in convos[1:]]
    else:
        convos = []
    
    chunks = []
    current_chunk = []
    current_length = len(overall_header)
    def finalize_chunk(chunk_list):
        return overall_header + "\n\n".join(chunk_list)
    for convo in convos:
        if len(convo) + len(overall_header) > max_chars:
            if current_chunk:
                chunks.append(finalize_chunk(current_chunk))
                current_chunk = []
                current_length = len(overall_header)
            subchunks = split_conversation_by_qna(overall_header + convo, max_chars)
            for sub in subchunks:
                chunks.append(sub)
        else:
            extra = 2 if current_chunk else 0
            if current_length + len(convo) + extra > max_chars and current_chunk:
                chunks.append(finalize_chunk(current_chunk))
                current_chunk = []
                current_length = len(overall_header)
                extra = 0
            current_chunk.append(convo)
            current_length += len(convo) + extra
    if current_chunk:
        chunks.append(finalize_chunk(current_chunk))
    return chunks

################################################################################
# Fallback Line-by-Line Splitter (if needed)
################################################################################

def split_tana_paste(tana_text, max_chars=MAX_FILE_SIZE):
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
# Manual Chat Conversion
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
        tana += "- #chatgptquestion\n"
        tana += "  - question::\n" + format_multiline_as_nodes(question_text, indent_level=6) + "\n"
        if question_time:
            tana += f"  - question_time:: {question_time}\n"
        tana += "  - chat_type:: text\n"
        tana += "  - answer::\n" + format_multiline_as_nodes(answer_text, indent_level=6) + "\n\n"
    return tana

################################################################################
# Best-Fit Decreasing Merging (Bin Packing)
################################################################################

def best_fit_decreasing(chunks, max_chars=MAX_FILE_SIZE):
    """
    Merge chunks using a best-fit decreasing strategy.
    Sort chunks by descending size, then place each chunk into the bin with the least
    remaining capacity that can still fit it. If none can, start a new bin.
    """
    prefix = "%%tana%%\n\n"
    chunk_info = []
    for c in chunks:
        if c.startswith(prefix):
            content = c[len(prefix):]
        else:
            content = c
        chunk_info.append((content, len(content)))
    
    chunk_info.sort(key=lambda x: x[1], reverse=True)
    
    bins = []  # each bin: (content, used_size)
    for content, size in chunk_info:
        best_idx = -1
        best_leftover = max_chars + 1
        for i, (bin_content, used) in enumerate(bins):
            leftover = max_chars - used
            if size + 2 <= leftover:
                if leftover - (size + 2) < best_leftover:
                    best_leftover = leftover - (size + 2)
                    best_idx = i
        if best_idx >= 0:
            bin_text, bin_used = bins[best_idx]
            bins[best_idx] = (bin_text + "\n\n" + content, bin_used + size + 2)
        else:
            bins.append((content, size))
    
    final_files = [prefix + b[0] for b in bins]
    return final_files

################################################################################
# File Statistics
################################################################################

def get_file_stats(files):
    if not files:
        return "No files generated"
    total_size = sum(len(f) for f in files)
    total_qna = sum(f.count("#chatgptquestion") for f in files)
    avg = total_size / len(files)
    stats = [
        f"**Total Files**: {len(files)}",
        f"**Total Size**: {total_size:,} characters",
        f"**Total Q&A Pairs**: {total_qna}",
        f"**Average File Size**: {avg:,.0f} characters"
    ]
    for i, f in enumerate(files):
        qna_count = f.count("#chatgptquestion")
        util = (len(f)/MAX_FILE_SIZE)*100
        stats.append(f"- File {i+1}: {len(f):,} chars ({util:.1f}% full), {qna_count} Q&A pairs")
    return "\n".join(stats)

################################################################################
# Streamlit App
################################################################################

st.set_page_config(page_title="ChatGPT to Tana Paste Converter", layout="wide")
st.title("ChatGPT to Tana Paste Converter")
st.write(
    "Upload your JSON file and convert it into Tana Paste.\n\n"
    "This version merges all assistant/tool messages between user messages into one answer,\n"
    "and splits the output by complete conversation nodes so that each file starts with the full conversation header\n"
    "and ends with a complete #chatgptquestion block. If a conversation node is too long, it will be split by Q–A blocks,\n"
    "with its header repeated in subsequent chunks. Full Q–A blocks are never cut between files.\n"
    "File names in the ZIP include the character length of each file."
)

conversion_mode = st.radio("Select conversion mode:", ["JSON File", "Manual Chat"])
file_prefix = st.sidebar.text_input("File name prefix", value="tana_paste")

# Use session_state to store conversion results so the preview selection doesn't reset.
if "merged_chunks" not in st.session_state:
    st.session_state.merged_chunks = None

if conversion_mode == "JSON File":
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    if uploaded_file is not None:
        raw_json_str = uploaded_file.read().decode("utf-8", errors="replace")
        try:
            json_data = json.loads(raw_json_str)
            st.success(f"Loaded JSON with {len(json_data)} conversations.")
            st.session_state.json_data = json_data
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
    
    if st.button("Convert JSON"):
        if not st.session_state.get("json_data"):
            st.error("No JSON loaded or file is empty.")
        else:
            progress_bar = st.progress(0)
            with st.spinner("Converting JSON..."):
                tana_text = json_to_tana_paste_merged_fast(st.session_state.json_data, progress_bar)
                progress_bar.progress(1.0)
                time.sleep(0.5)
            with st.spinner("Splitting into chunks..."):
                initial_chunks = split_tana_by_conversations(tana_text, max_chars=MAX_FILE_SIZE)
                merged_chunks = best_fit_decreasing(initial_chunks, max_chars=MAX_FILE_SIZE)
            if merged_chunks:
                st.session_state.merged_chunks = merged_chunks
                st.subheader("Conversion Statistics")
                st.markdown(get_file_stats(merged_chunks))
            else:
                st.error("No content generated.")
                
if st.session_state.merged_chunks:
    preview_index = st.selectbox(
        "Select file to preview:",
        range(len(st.session_state.merged_chunks)),
        format_func=lambda i: f"File {i+1} ({len(st.session_state.merged_chunks[i]):,} chars)",
        key="preview_index"
    )
    st.subheader("Preview of Selected File")
    st.text_area("Preview", value=st.session_state.merged_chunks[preview_index], height=400)
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i, content in enumerate(st.session_state.merged_chunks):
            fname = f"{file_prefix}_part_{i+1}_{len(content)}chars.txt"
            zipf.writestr(fname, content)
    st.download_button("Download ZIP", data=zip_buffer.getvalue(), file_name=f"{file_prefix}.zip", mime="application/zip")
                
elif conversion_mode == "Manual Chat":
    manual_text = st.text_area("Paste your chat transcript here", height=400,
                               help="Blocks with 'You said:' and 'ChatGPT said:' markers.")
    if st.button("Convert Manual Chat"):
        if not manual_text.strip():
            st.error("Please paste some chat text first.")
        else:
            tana_text = manual_chat_to_tana_paste(manual_text)
            chunks = split_tana_by_conversations(tana_text, max_chars=MAX_FILE_SIZE)
            final_files = best_fit_decreasing(chunks, max_chars=MAX_FILE_SIZE)
            if final_files:
                st.subheader("Tana Paste Output Statistics")
                st.markdown(get_file_stats(final_files))
                st.subheader("Preview of First File")
                st.text_area("Preview", value=final_files[0], height=400)
                zbuf = BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for i, fc in enumerate(final_files):
                        fname = f"{file_prefix}_manual_part_{i+1}_{len(fc)}chars.txt"
                        zipf.writestr(fname, fc)
                st.download_button("Download ZIP", data=zbuf.getvalue(), file_name=f"{file_prefix}_manual.zip", mime="application/zip")
            else:
                st.error("No valid content produced.")

st.markdown("---")
st.caption("ChatGPT to Tana Paste Converter | Made with ❤️")
