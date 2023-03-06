# Usage: ./strip_epoch_weights my_results_dir (this removes all .pth files with "epoch" in their file name)
# Usage: ./strip_epoch_weights my_results_dir -a (this removes ALL .pth files)

set -e # exit when any command fails

dir_name="${1}"

ALL="${2:-"epoch"}"

if [ "$ALL" == "-a" ]; then
  PATTERN=".pth"
else
  PATTERN="*epoch*.pth"
fi

mkdir -p big_results
cp -r "$dir_name" big_results/"$dir_name"
find "$dir_name" -type f -name "$PATTERN" -delete
