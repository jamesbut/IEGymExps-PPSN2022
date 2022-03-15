#!/bin/bash

# Read server names as csv
IFS=',' read -ra server_names <<< "$1"
for server_name in "${server_names[@]}"; do

    server_path="~/IEGymExps"
    data_dir_name=$server_name"_data"
    zip_file_name=$data_dir_name".zip"

    # Zip data on server
    ssh $server_name "cd $server_path; zip -r $zip_file_name data"

    # Transfer data from server
    scp $server_name:$server_path"/"$zip_file_name ../data

    # Unzip server data and rename
    cd ../data
    unzip $zip_file_name
    mv data $data_dir_name

    # Delete zip files
    rm $zip_file_name
    ssh $server_name "cd $server_path; rm $zip_file_name"

done
