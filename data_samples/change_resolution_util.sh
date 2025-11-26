INPUT_DIR="clips_no_audio"
OUTPUT_DIR="clips_compressed"

mkdir -p "$OUTPUT_DIR"

for in_path in "$INPUT_DIR"/*; do
    filename=$(basename "$in_path")
    out_path="$OUTPUT_DIR/$filename"

    ffmpeg -y -i "$in_path" \
        -vf "scale=224:224" \
        -c:v libx264 -crf 23 -preset medium \
        -c:a copy \
        "$out_path"
done