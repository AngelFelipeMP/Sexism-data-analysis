#!/bin/sh

ebs_path="/dev/nvme2n1"

# S3-bucket
sudo apt update -y
sudo apt install s3fs -y
mkdir s3-bucket

# EBS volume
sudo file -s $ebs_path
sudo mkfs -t xfs $ebs_path
mkdir ebs-volume

## SSD internal disck
mkdir ssd-volume
