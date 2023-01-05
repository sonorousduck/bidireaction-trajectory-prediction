# Connecting to Go with proxy internet

Tools required: Squid

Here is the URL that will pretty much explain everything important. Connecting to the GPU is the more tricky thing that I will explain.

https://unix.stackexchange.com/questions/116191/give-server-access-to-internet-via-client-connecting-by-ssh

### Connecting to Go GPU with proxy internet

Tools required: Squid

This is slightly more tricky as we need to proxy to the Raspberry Pi, then proxy to the GPU. In the previous step, the proxy should have been set up on the raspberry pi. Pay attention to the URL where it talks about setting it up if your Host A currently uses a proxy. You will need to add this piece, then since you are using squid again to proxy from Raspberry Pi to GPU, you will need to change the HTTP_Port from 3129 to something else (I chose 3130)

You then can use the following command (if you chose 3130)

```
ssh -R 3130:localhost:3129 <gpu_ip>
```

For the Go1 GPU
```
ssh -R 3130:localhost:3129
```


Then, run
```
source /etc/environment
```
where the two export variables should be stored (from instructions above)

This should give you access to the internet. However, due to the multi-tunnel, I was running into certificate issues. While this isn't the best, you can bypass certificate issues with the following commands:

If using apt
```
sudo apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false install/upgrade
```

If using pip
```
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

These allowed me to install packages and update/upgrade.

 -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false 


 ### Installing Squid:
 Windows has an MSI installer, Linux it is a apt package. Mac has it as a brew package


### SSH from Local Computer to Raspberry Pi
```
    ssh -R 3129:localhost:3128 pi@192.168.12.1
```

 ### SSH from Raspberry Pi to GPU:
 ```
    ssh -R 3131:localhost:3130 gpu_main
    ssh -R 3131:localhost:3130 gpu_head
    ssh -R 3131:localhost:3130 gpu_side
 ```

 Or, I set up aliases for each for convenience
 ```
    gpu_head_ssh
    gpu_side_ssh
    gpu_main_ssh
 ```


 ### Additional Setup things I had to do

 In /etc/apt/apt.conf.d/99verify-peer.conf, I had to add
 ```
 Acquire { https::Verify-Peer false }
 ```

 as a one time thing for each one set up


 # Easy Instructions

 I set up a bunch of convenient aliases to abstract away the difficulty of doing this.

 Step 1) Set up Squid on your computer

 Step 2) SSH to raspberry pi on Go1 with this command:
 ```
    ssh -R 3129:localhost:3128 pi@192.168.12.1
 ```

 Step 3) On the Raspberry Pi, you then can SSH to any of the GPUs using the following alias (Just type it in as written below)

For Main GPU:
```
    gpu_main_ssh
```

For Side GPU:
```
    gpu_side_ssh
```

For Head GPU:
```
    gpu_head_ssh
```