def parse_file(filename):
    metadata = filename.split("_")

    video_id = metadata[0]

    item = {
        "filename": filename,
        "video_id": video_id,
    }

    temp_mix = metadata[1]

    if temp_mix == "mix":
        item.update({
            "mix": 1,
            "emotion_1": metadata[2],
            "emotion_2": metadata[3],
            "emotion_1_salience": metadata[4],
            "emotion_2_salience": metadata[5],
            "version": metadata[6][3]
        })
    else:
        item.update({
            "mix": 0,
            "emotion_1": metadata[1],
            "version": metadata[3][3]
        })

        sit_int = metadata[2]
        if sit_int[0:3] == "int":
            item["intensity_level"] = sit_int[3]
        elif sit_int[0:3] == "sit":
            item["situation"] = sit_int[3]

    return item


if __name__ == "__main__":
    # Example usage
    f = "A102_ang_int1_ver1"
    parsed_data = parse_file(f)
    print(parsed_data)

    f = "A438_mix_disg_hap_30_70_ver1"
    parsed_data = parse_file(f)
    print(parsed_data)