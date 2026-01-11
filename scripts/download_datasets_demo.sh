mkdir ../dataset
hf download jsnzwu/MoFlow-demo FC_demo_30f_fp16.zip --repo-type dataset --local-dir ../dataset/
hf download jsnzwu/MoFlow-demo DT_demo_30f_fp16.zip --repo-type dataset --local-dir ../dataset/
unzip ../dataset/FC_demo_30f_fp16.zip -d ../dataset/
rm ../dataset/FC_demo_30f_fp16.zip
unzip ../dataset/DT_demo_30f_fp16.zip -d ../dataset/
rm ../dataset/DT_demo_30f_fp16.zip