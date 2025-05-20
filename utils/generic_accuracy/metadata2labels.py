from utils.filename_parser import parse_filename


def metadata2label(metadata):

    ret = {}

    filename = metadata.get("filename", None)
    emotion_1 = metadata.get("emotion_1", None)
    emotion_2 = metadata.get("emotion_2", None)

    if emotion_2:
        emotion_1_salience = metadata["emotion_1_salience"]
        emotion_2_salience = metadata["emotion_2_salience"]

        ret[filename] = [
            {"emotion": emotion_1, "salience": emotion_1_salience},
            {"emotion": emotion_2, "salience": emotion_2_salience}
        ]
    else:
        ret[filename] = [
            {"emotion": emotion_1, "salience": 1.0}
        ]
    return ret


def main():
    # Example usage
    f = "A102_ang_int1_ver1"
    parsed_data = parse_filename(f)
    print(parsed_data)

    m = metadata2label(parsed_data)
    print(m)

    f = "A438_mix_disg_hap_30_70_ver1"
    parsed_data = parse_filename(f)
    print(parsed_data)

    m = metadata2label(parsed_data)
    print(m)

if __name__ == "__main__":
    main()