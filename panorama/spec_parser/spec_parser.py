import re
import os
import json


def parse_file(file_path):
    result = {
        "id": os.path.basename(file_path).split(".")[0],
        "keyCount": 0,
        "sentenceCount": 0,
        "items": [],
    }

    with open(file_path, "r", encoding="utf-8") as file:
        content = "\n" + file.read()

    pattern = r"(?:\[(\d{4})\]|\n\((\d+)\))(.*?)(?=\[\d{4}\]|\n\(\d+\)|$)"

    matches = re.findall(pattern, content, re.DOTALL)

    key_counts = {}
    sentence_count = 0

    for bracket_key, paren_key, content_text in matches:
        original_key = bracket_key if bracket_key else paren_key
        sentence_pattern = r"[.!?។።፡。۔؟።!\?]+[\s]*"
        sentences = re.split(sentence_pattern, content_text.strip())
        sentences = [s for s in sentences if s.strip()]
        sentence_count += len(sentences)

        if original_key in key_counts:
            key_counts[original_key] += 1
            final_key = f"{key_counts[original_key]}-{original_key}"
        else:
            key_counts[original_key] = 1
            final_key = f"1-{original_key}"

        item = {"key": final_key, "content": content_text.strip()}
        result["items"].append(item)

    result["keyCount"] = len(result["items"])
    result["sentenceCount"] = sentence_count

    return result


def save_parsed_file(file_path, parsed_data):
    dir_path = os.path.dirname(file_path)

    parsed_dir = os.path.join(dir_path, "parsed")

    if not os.path.exists(parsed_dir):
        os.makedirs(parsed_dir)

    file_name = os.path.basename(file_path)

    file_name_base = os.path.splitext(file_name)[0]

    new_file_path = os.path.join(parsed_dir, f"{file_name_base}_parsed.json")

    json_str = json.dumps(parsed_data, ensure_ascii=False, indent=2)

    with open(new_file_path, "w", encoding="utf-8") as file:
        file.write(json_str)

    return new_file_path


def processor(file_path):
    parsed_data = parse_file(file_path)

    output_path = save_parsed_file(file_path, parsed_data)

    print("-" * 50)
    print(f"{file_path} parsed!")
    print(f"{len(parsed_data['items'])} items parsed!")


def calculate_average(parsed_file_folder_path):
    key_count_sum = 0
    sentence_count_sum = 0

    parsed_data_list = []
    for file_name in os.listdir(parsed_file_folder_path):
        if file_name.endswith("_parsed.json"):
            with open(
                os.path.join(parsed_file_folder_path, file_name), "r", encoding="utf-8"
            ) as file:
                parsed_data = json.load(file)
            parsed_data_list.append(parsed_data)

    key_count_sum = sum(parsed_data["keyCount"] for parsed_data in parsed_data_list)
    sentence_count_sum = sum(
        parsed_data["sentenceCount"] for parsed_data in parsed_data_list
    )

    average_key_count = key_count_sum / len(parsed_data["items"])
    average_sentence_count = sentence_count_sum / len(parsed_data["items"])

    return average_key_count, average_sentence_count


if __name__ == "__main__":
    folder_path = "data/text"
    txt_files = []

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        exit()

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            txt_files.append(os.path.join(folder_path, filename))

    for txt_file in txt_files:
        processor(txt_file)

    print("-" * 50)
    print("All files parsed!")

    average_key_count, average_sentence_count = calculate_average("data/text/parsed")
    print(f"Average key count: {average_key_count}")
    print(f"Average sentence count: {average_sentence_count}")
