import argparse
parser = argparse.ArgumentParser(description='Select task to execute.')
# choices=['attack',"defense","test",'visualize']
parser.add_argument('--task', type=str, default="attack", required=False, help='the task which is selected to execute,such as ')
# parser.add_argument('--item', type=str, choices=['generate backdoor samples', ], required=True, help='A specific item under the task')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

"""
Users can select the task to execute by passing parameters, suach as 
1. The task of generating and showing backdoor samples
    python test_BadNets.py --task "generate backdoor samples"

2. The task of showing backdoor samples
    python test_BadNets.py --task "show train backdoor samples"
    python test_BadNets.py --task "show test backdoor samples"
    
3. The task of training backdoor model
    python test_BadNets.py --task "attack"

4.The task of testing backdoor model
    python test_BadNets.py --task "test"

5.The task of generating latents
    python test_BadNets.py --task "generate latents"

6.The task of visualizing latents by t-sne
    python test_BadNets.py --task "visualize latents by t-sne"
    python test_BadNets.py --task "visualize latents for target class by t-sne"

"""

