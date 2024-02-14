import re
from nltk.tokenize import sent_tokenize, word_tokenize

def insert_newlines(text):
    formatted_text = re.sub(r'(?<!\n)(#\w+-\d+#:)', r'\n\1', text)
    return formatted_text

def extract_query(txt):
    queries = []
    # Find the pattern with #Query-#
    pattern = r"Query-\d+#:\s*(.+?)(?=\s*#|$)|Query#:\s*(.+?)(?=\s*#|$)"
    matches = re.findall(pattern, txt)
    for match in matches:
        # Concatenate tuples from re.findall, if any
        combined_match = ''.join(match)
        if '[' in combined_match:
            # Find the last occurrence of ']'
            last_bracket_pos = combined_match.rfind(']')
            # Find the corresponding '['
            first_bracket_pos = combined_match.rfind('[', 0, last_bracket_pos)

            # Split the string into two parts
            first_query = combined_match[:first_bracket_pos].strip()
            second_query = combined_match[first_bracket_pos+1:last_bracket_pos].strip()

            queries.append(first_query)
            queries.extend(second_query.split('; '))
        else:
            queries.append(combined_match)
    return queries

def clean_query(text):
#     # Function to remove only the content in the last bracket of each query
#     def remove_last_bracket_and_space(match):
#         content = match.group()
#         last_bracket_pos = content.rfind(']')
#         first_bracket_pos = content.rfind('[', 0, last_bracket_pos)
#         if first_bracket_pos != -1 and last_bracket_pos != -1:
#             return content[:first_bracket_pos].rstrip() + content[last_bracket_pos+1:]
#         return content
#     # Apply the function to each line that starts with #Query
#     cleaned_text = re.sub(r'#Query-.*?:.*', remove_last_bracket_and_space, text)
    # Remove lines that start with #Thought-xx and #Done#
    cleaned_text = re.sub(r'#Thought-\d+#:.*|Thought-\d+#:.*|#Done#', '', text, flags=re.MULTILINE)
    # Remove lines that do not start with #Query or #Knowledge
    cleaned_text = re.sub(r'^(?!#Query|#Knowledge|Query|Knowledge).*$', '', cleaned_text, flags=re.MULTILINE)
    # Remove excessive newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()

def split_summary_into_parts(summary, word_limit = 30):
    """
    Splits the summary into parts with less than word_limit words each.
    
    :param summary: A string containing the summary.
    :return: A list of strings, where each string is a part of the summary with less than 100 words.
    """
#     nltk.download('punkt')  # Ensure necessary NLTK data is downloaded

    sentences = sent_tokenize(summary.strip())
    summary_parts = []
    current_part = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(word_tokenize(sentence))
        if current_word_count + word_count > word_limit:
            # If adding this sentence exceeds word_limit words, process the current part
            summary_parts.append(' '.join(current_part))
            current_part = [sentence]
            current_word_count = word_count
        else:
            # Otherwise, add this sentence to the current part
            current_part.append(sentence)
            current_word_count += word_count

    # Add the last part if it's not empty
    if current_part:
        summary_parts.append(' '.join(current_part))

    return summary_parts