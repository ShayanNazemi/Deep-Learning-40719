#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# create a cpu device if cuda is not available or cuda_device=None otherwise
# create a cuda:{cuda_device} device.
#################################################################################
cuda_device = torch.cuda.current_device()
device = torch.cuda.device(cuda_device)
pass
#################################################################################
#                                   THE END                                     #
#################################################################################
print(device)





batch_size = 32
#################################################################################
#                          COMPLETE THE FOLLOWING SECTION                       #
#################################################################################
# Initialize and download trainset and testset with datasets.FashionMNIST and
# transform data into torch.Tensor. Initialize trainloader and testloader with
# given batch_size.
#################################################################################

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform)

testset = datasets.FashionMNIST(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

#################################################################################
#                                   THE END                                     #
#################################################################################
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')