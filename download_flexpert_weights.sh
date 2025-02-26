echo "Downloading Flexpert weights..."

# Create the directory if it doesn't exist
mkdir -p models/weights

# Set file information for Flexpert weights
WEIGHTS_URL-3d="https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_3d_weights.bin"
OUTPUT_FILE-3d="models/weights/flexpert_3d_weights.bin"

WEIGHTS_URL-Seq="https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_seq_weights.bin"
OUTPUT_FILE-Seq="models/weights/flexpert_seq_weights.bin"

echo "Downloading Flexpert-3D weights..."
wget --no-check-certificate "${WEIGHTS_URL-3d}" -O ${OUTPUT_FILE-3d}

echo "Downloading Flexpert-Seq weights..."
wget --no-check-certificate "${WEIGHTS_URL-Seq}" -O ${OUTPUT_FILE-Seq}

echo "Flexpert weights download completed."
