import argparse
import sys

import torch
import torch.nn.functional as F
from torch import nn

from torch import optim

from data import mnist
from model import MyAwesomeModel
import model


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        my_model = MyAwesomeModel(128, 10)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(my_model.parameters(), lr=0.001)
        trainloader, testloader = mnist()
        model.train(my_model, trainloader, testloader, criterion, optimizer, epochs=5)

        checkpoint = {'hidden_size': 128,
              'output_size': 10,
              'state_dict': my_model.state_dict()}

        torch.save(checkpoint, 'checkpoint.pth')

        
    def evaluate(self):

        def load_checkpoint(filepath):
            checkpoint = torch.load(filepath)
            my_model = MyAwesomeModel(checkpoint['hidden_size'],
                                    checkpoint['output_size'])
            my_model.load_state_dict(checkpoint['state_dict'])
            
            return my_model
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        my_model = load_checkpoint('checkpoint.pth')
        _, testloader = mnist()
        criterion = nn.NLLLoss()

        test_loss, accuracy = model.validation(my_model, testloader, criterion)
        print(f'test loss: {test_loss}, accuracy: {accuracy}')
   



if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    