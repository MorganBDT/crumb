max_run_ind=4
dir_name="${1}"
dir_structure="${2-"iid/Crumb_SqueezeNet_offline"}" # include no slashes / at beginning or end of this
mkdir -p "$dir_name"/"$dir_structure"
for i in `seq 0 $max_run_ind`
do
    mv "$dir_name"_run"$i"/"$dir_structure"/log.log      "$dir_name"_run"$i"/"$dir_structure"/runs-"$i"/log.log
    mv "$dir_name"_run"$i"/"$dir_structure"/runs-"$i"    "$dir_name"/"$dir_structure"/runs-"$i"
done