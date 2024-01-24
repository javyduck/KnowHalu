import re

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
    cleaned_text = re.sub(r'#Thought-\d+#:.*|Thought-\d+#:.*', '', text, flags=re.MULTILINE)
    # Remove lines that do not start with #Query or #Knowledge
    cleaned_text = re.sub(r'^(?!#Query|#Knowledge|Query|Knowledge).*$', '', cleaned_text, flags=re.MULTILINE)
    # Remove excessive newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()