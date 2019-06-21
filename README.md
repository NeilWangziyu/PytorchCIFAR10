# PytorchCIFAR10

Clone and modify from
https://github.com/kuangliu/pytorch-cifar.git

build two images carinapaas.azurecr.io/cgc/testcifar10_cpu and carinapaas.azurecr.io/cgc/testcifar10_gpu

The CIFAR10 dataset is included in images, if run on local, it would automatically downloaded.

Totally four parameters:

--GPU True/False

--model default:resnet, choose from resnet, vgg, googlenet, densenet,mobilenet,mobilenet2

--epoch default:200

--learningRate default:0.1

On local(suzlab1080-012):
CPU: 
` ` ` bash
python3.6 /pytorch_train/bench_mark_GPU/main.py --GPU True --model resnet --epoch 200 
` ` `

GPU: 
` ` ` bash
python3.6 /pytorch_train/bench_mark_CPU/main.py --GPU False --model resnet --epoch 200
` ` `


On Docker:
In docker image, running the code would be use the default parameter
For GPU test:  
` ` ` bash
docker run carinapaas.azurecr.io/cgc/testcifar10_gpu 
` ` `

For CPU test:  
` ` ` bash
docker run carinapaas.azurecr.io/cgc/testcifar10_cpu 
` ` `


On CAP:
**GPU**
{ 

  "id": "276134e5-6c70-42a2-9aaf-e1ef9b6711e6", 

  "name": "bench_mark_GPU", 

  "owner": "admin", 

  "submittedOn": "636964724255037548", 

  "priority": 1, 

  "runningTimeout": "4320000", 

  "waitingTimeout": "43200", 

  "status": "FINISHED", 

  "graph": { 

    "id": "dba867e9-179e-48c4-8f09-cd0ba5ffa5e7", 

    "nodes": [ 

      { 

        "id": "node-2", 

        "module": { 

          "name": " bench_mark_GPU", 

          "description": "train a pytorch resnet model for cifar10 classification using GPU", 

          "owner": "admin", 

          "familyId": "bba01e71-22a9-45e2-88b4-7c4a2d47aeb4", 

          "version": 1, 

          "submittedOn": "636964235917351819", 

          "type": "DOCKER", 

          "dockerSpec": { 

            "resource": { 

              "cpu": 4, 

              "gpu": 1, 

              "memory": 4096 

            }, 

            "command": "python /app/main.py --GPU True --model resnet --epoch 200", 

            "image": "carinapaas.azurecr.io/cgc/testcifar10_gpu" 

          }  

        }, 

        "runtimeInfo": { 

          "forceRerun": true 

        }, 

        "layout": { 

          "x": 329.824554, 

          "y": 265, 

          "width": 225, 

          "height": 34 

        } 

      } 

    ] 

  } 

} 
**CPU**
{ 

  "id": "276134e5-6c70-42a2-9aaf-e1ef9b6711e6", 

  "name": " bench_mark_CPU", 

  "owner": "admin", 

  "submittedOn": "636964724255037548", 

  "priority": 1, 

  "runningTimeout": "4320000", 

  "waitingTimeout": "43200", 

  "status": "FINISHED", 

  "graph": { 

    "id": "dba867e9-179e-48c4-8f09-cd0ba5ffa5e7", 

    "nodes": [ 

      { 

        "id": "node-2", 

        "module": { 

          "name": " bench_mark_CPU", 

          "description": " train a pytorch resnet model for cifar10 classification using CPU", 

          "owner": "admin", 

          "familyId": "bba01e71-22a9-45e2-88b4-7c4a2d47aeb4", 

          "version": 1, 

          "submittedOn": "636964235917351819", 

          "type": "DOCKER", 

          "dockerSpec": { 

            "resource": { 

              "cpu": 4, 

              "gpu": 0, 

              "memory": 4096 

            }, 

            "command": "python /app/main.py --GPU False --model resnet --epoch 200", 

            "image": "carinapaas.azurecr.io/cgc/testcifar10_cpu" 

          }  

        }, 

        "runtimeInfo": { 

          "forceRerun": true 

        }, 

        "layout": { 

          "x": 329.824554, 

          "y": 265, 

          "width": 225, 

          "height": 34 

        } 

      } 

    ] 

  } 

} 