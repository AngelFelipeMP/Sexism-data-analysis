#!/bin/sh

# ebs_path="/dev/nvme1n1"
# ssd_path="/dev/nvme2n1"

# df -h
# echo " "
# lsblk

## s3-bucket nount
s3fs angel-ec2-bucket s3-bucket -o iam_role=auto

# ## External EBS volume 
# sudo mount $ebs_path ebs-volume
# ##sudo chmod 777 -R /ebs-volume
# sudo chown ubuntu:ubuntu ebs-volume

# ## SSD internal disck
# sudo mkfs -t xfs $ssd_path
# sudo mount $ssd_path ssd-volume
# sudo chown ubuntu:ubuntu ssd-volume

# echo " "
# df -h

# source activate pytorch