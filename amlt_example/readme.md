# Here is a small sample config to get started running jobs

# step 1: PIM elevation (have to do this every 8 hours)
elevate your PIM access [here](https://ms.portal.azure.com/#blade/Microsoft_Azure_PIMCommon/ActivationMenuBlade/azurerbac) for gcrllama2ws, msrresrchvc
	- optionally also activate: Deep Learning Group, huashanvc1

# set up project
`amlt project create example_project internblobdl`

# add some workspace
`amlt workspace add gcrllama2ws --resource-group GCRllama2 --subscription ca45784d-7313-48b5-bf5e-3ae34d936a7a`

# run a job using config file
- edit `example_job.yaml`
  - change "container_name: t-chansingh" to "container_name: <your_container_name>"
  - change "local_dir: /home/chansingh/cookiecutter-ml-research/amlt_example" to the path where this directory is located on your machine

`amlt run example_job.yaml`

# note
- if you want to debug your jobs, you can put a sleep command in your config file (e.g. replace `python example_job_script.py ` with `sleep 3600`)
- then, when you run the job, it will print a PORTAL URL to monitor the job
  - if you click on this link, there is an icon called "Debug and monitor", if you click on this it will give you a command that lets you ssh in the job and interactively try things out