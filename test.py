"""
There are N boxes and you start in box 0.  Each box has a distinct number in 0, 1, 2, ..., N-1, and 
each box may have some keys to access the next box. 
Each box i has a list of keys boxes[I], and each key boxes[I][j] is an integer in [0, 1, ..., N-1] where N = boxes.length.  
A key boxes[I][j] = p opens the room with number p.
Initially, all the boxes start closed (except for box 0). 
You can walk back and forth between boxes freely.

Input: [[1],[2],[3],[]]
Output: true

Input: [[1],[2],[],[]]
Output: false
"""

boxes={}

result_hash = {
    0: True,
    1: True,
    2: True,
    3: False
}
[[1],[2],[3],[]]

def visited(input_array):
    my_result_hash = {}
    no_boxes = len(input_array)
    for i in range(no_boxes):
        my_result_hash[i] = False
    
    for position, box in enumerate(input_array):
        if box[0]:
            my_key = box[0]
            my_result_hash[position] = True
    
    if False in my_result_hash.values():
        return False

    return True

