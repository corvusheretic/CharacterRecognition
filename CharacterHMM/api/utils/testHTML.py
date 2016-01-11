'''
Created on Jan 1, 2016

@author: kalyan
'''

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from collections import Counter

from api.utils.HTML import Table, TableCell, TableRow

import unittest
import pdb

def db_StallExec(choice):
    if(choice):
        pdb.set_trace()

resultList=[('A',['o', 'A', 'o', 'A', 'A', 'A', 'A', 'A', 'o', 'o', 'A', 'o', 'O', 'o', 'o', 'e', 'o', 'A', 'A', 'A']),
('B',['B', 'o', 'D', 'B', 'B', 'B', 'Q', 'B', 'B', 'B', 'D', 'B', 'z', 'e', 'B', 'z', 'B', 'B', 'x', 'R']),
('C',['c', 'C', 'c', 'C', 'a', 'C', 'C', 'c', 'C', 'C', 'o', 'c', 'c', 'C', 'c', 'c', 'C', 'C', 'C', 'C']),
('D',['D', 'c', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'o', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']),
('E',['B', 'E', 'e', 'E', 'O', 'o', 'J', 'E', 'e', 'R', 'e', 'E', 'o', 'E', 'B', 'E', 'E', 'E', 'E', 'E']),
('F',['F', 'o', 'F', 'p', 'o', 'p', 'e', 'T', 'o', 'A', 'o', 'p', 'p', 'p', 'F', 'F', 'o', 'P', 'p', 'F']),
('G',['o', 'o', 'B', 'O', 'G', 'o', 'G', 'G', 'c', 'G', 'O', 'c', 'O', 'G', 'G', 'Q', 'a', 'G', 'Q', 'C']),
('H',['H', 'e', 'o', 'H', 'H', 'H', 'e', 'H', 'H', 'e', 'o', 'e', 'H', 'H', 'H', 'H', 'o', 'H', 'U', 'U']),
('I',['o', 'q', 'x', 'o', 'd', 'o', 'c', 'o', 'i', 'I', 'e', 'o', 'q', 'o', 'F', 'P', 'o', 'o', 'S', 's']),
('J',['o', 'e', 'B', 'U', 'y', 'A', 'e', 'g', 'U', 'U', 'c', 'e', 'D', 'P', 'y', 'C', 'D', 'Y', 'd', 'o']),
('K',['K', 'o', 'b', 'K', 'K', 'o', 'e', 'K', 'K', 'K', 'u', 'o', 'K', 'K', 'o', 'b', 'K', 'K', 'K', 'd']),
('L',['L', 'L', 'L', 'o', 'L', 'L', 'o', 'L', 'L', 'L', 'L', 'c', 'L', 'L', 'L', 'L', 'c', 'L', 'L', 'L']),
('M',['M', 'M', 'n', 'M', 'M', 'n', 'M', 'M', 'M', 'M', 'M', 'M', 'r', 'M', 'M', 'H', 'M', 'M', 'n', 'm']),
('N',['N', 'e', 'M', 'M', 'N', 'N', 'N', 'N', 'M', 'N', 'M', 'N', 'o', 'N', 'N', 'N', 'N', 'N', 'N', 'N']),
('O',['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G']),
('P',['T', 'p', 'F', 'p', 'p', 'p', 'e', 'T', 'F', 'p', 'r', 'r', 'p', 'z', 'P', 'p', 'p', 'T', 'D', 'D']),
('Q',['Q', 'o', 'O', 'Q', 'Q', 'Q', 'G', 'Q', 'e', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'R', 'o', 'Q', 'Q', 'Q']),
('R',['B', 'e', 'b', 'o', 'B', 'R', 'e', 'R', 'R', 'R', 'R', 'z', 'e', 'e', 'R', 'R', 'o', 'R', 'R', 'R']),
('S',['Z', 'O', 'D', 'Z', 'Z', 'R', 'e', 's', 'S', 'z', 'E', 'e', 'S', 'S', 'S', 'e', 'S', 'o', 'S', 'S']),
('T',['T', 'T', 'T', 'T', 'T', 'o', 'e', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'o', 'v', 'T', 'T', 'c', 'Y']),
('U',['o', 'U', 'U', 'o', 'U', 'U', 'c', 'U', 'U', 'v', 'U', 'U', 'U', 'U', 'U', 'H', 'U', 'U', 'U', 'U']),
('V',['o', 'Y', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'o', 'U', 'o', 'V', 'V', 'U', 'e', 'o', 'v', 'v']),
('W',['W', 'W', 'W', 'W', 'W', 'W', 'W', 'e', 'W', 'W', 'W', 'W', 'o', 'w', 'V', 'w', 'U', 't', 'U', 'o']),
('X',['o', 'X', 'X', 'o', 'x', 'o', 'c', 'X', 'o', 'X', 'X', 'A', 'X', 'X', 'X', 'o', 'o', 'o', 'X', 'X']),
('Y',['Y', 'e', 'Y', 'o', 'Y', 'Y', 'o', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'e', 'o', 'Y', 'Y', 'Y']),
('Z',['S', 'S', 'Z', 'Z', 'z', 'o', 'S', 'S', 'Z', 'z', 'z', 'o', 'R', 'Z', 'o', 'g', 'q', 'Z', 'S', 'S']),
('a',['a', 'a', 'E', 'a', 'R', 'a', 'a', 'a', 'a', 't', 'a', 'a', 'a', 'a', 'a', 'e', 'B', 'a', 'Q', 'o']),
('b',['L', 'e', 'D', 'e', 'e', 'L', 'e', 'u', 'b', 'e', 'e', 'D', 'b', 'b', 'b', 'o', 'b', 'b', 'b', 'D']),
('c',['c', 'C', 'C', 'c', 't', 'C', 'C', 'c', 'a', 'C', 'c', 'c', 'o', 'C', 'c', 'c', 'c', 'a', 'C', 'c']),
('d',['d', 'e', 'A', 'c', 'o', 'j', 'd', 'c', 'd', 'u', 'd', 'd', 'd', 'd', 'S', 'e', 'o', 'o', 'o', 'd']),
('e',['e', 'e', 'e', 'e', 't', 'e', 'S', 'e', 'e', 'e', 'o', 'z', 'e', 'e', 'e', 'E', 'S', 'e', 'Q', 'Q']),
('f',['A', 'o', 'e', 'p', 'o', 's', 'o', 'S', 'o', 'e', 'S', 'o', 'o', 'e', 'l', 'e', 'o', 'o', 'E', 'e']),
('g',['g', 'S', 'S', 'o', 'q', 'g', 'S', 'e', 'q', 'p', 'D', 'B', 'o', 'e', 'o', 'B', 'o', 'o', 'E', 'B']),
('h',['o', 'o', 'e', 'h', 'c', 'o', 'H', 'o', 'h', 'b', 'o', 'e', 'M', 'h', 'h', 'o', 'o', 'H', 'c', 'e']),
('i',['j', 'L', 'L', 'a', 'u', 'L', 'Y', 'o', 'o', 'i', 'o', 'o', 'o', 'o', 'o', 'J', 'o', 'L', 'a', 'U']),
('j',['o', 'e', 'e', 'o', 'c', 'o', 'e', 'o', 'e', 'J', 'o', 'e', 'e', 'e', 'U', 'e', 'o', 'o', 'o', 'e']),
('k',['K', 'K', 'o', 'K', 'o', 'K', 'o', 'b', 'k', 'k', 'u', 'e', 'e', 'K', 'o', 'K', 'd', 'K', 'K', 'k']),
('l',['l', 'I', 'o', 'o', 'o', 's', 'e', 'P', 'o', 'e', 'e', 'e', 'n', 'o', 'B', 'H', 'e', 'o', 'Q', 'o']),
('m',['o', 'n', 'o', 'M', 'm', 'n', 'm', 'm', 'm', 'c', 'M', 'M', 'm', 'o', 'M', 'o', 'm', 'o', 'n', 'n']),
('n',['o', 'n', 'n', 'n', 'n', 'r', 'n', 'm', 'n', 'o', 'n', 'n', 'n', 't', 'n', 'H', 'n', 'o', 'n', 'n']),
('o',['O', 'O', 'O', 'O', 'E', 'O', 'O', 'o', 'D', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'D', 'O', 'o']),
('p',['n', 'p', 'p', 'p', 'v', 'T', 'p', 'e', 'p', 'p', 'r', 'p', 'p', 'P', 'p', 'p', 'p', 'T', 'F', 'D']),
('q',['o', 'o', 'o', 'o', 'z', 'o', 'q', 'o', 'q', 'Y', 'q', 'o', 'n', 'o', 'z', 'q', 'a', 'o', 'z', 'g']),
('r',['r', 'T', 'p', 'T', 'n', 'Y', 'p', 's', 'r', 'n', 'o', 'r', 'o', 'R', 'p', 'p', 'e', 'q', 'P', 'p']),
('s',['e', 'S', 'B', 'S', 'A', 'e', 'S', 's', 'Z', 's', 'q', 'o', 'z', 'E', 's', 'e', 'S', 'o', 'S', 'Z']),
('t',['t', 'e', 'L', 't', 'o', 'D', 'o', 'o', 't', 'y', 'b', 'x', 'o', 'e', 'o', 'H', 'e', 'q', 't', 'o']),
('u',['u', 'u', 'U', 'U', 'u', 'u', 't', 'u', 'U', 'u', 'U', 'U', 'u', 'U', 'U', 'H', 'h', 'u', 'U', 'U']),
('v',['V', 'V', 'e', 'v', 'v', 'V', 'V', 'v', 'V', 'o', 'V', 'v', 'v', 'v', 'V', 'V', 'U', 'v', 'v', 'v']),
('w',['W', 'o', 'W', 'u', 'v', 'W', 'W', 'W', 'W', 'W', 'w', 'W', 'W', 'W', 'W', 'w', 'u', 'W', 'u', 'w']),
('x',['o', 'X', 'X', 'x', 'K', 'X', 'X', 'X', 'X', 'X', 'a', 'x', 'X', 'd', 'P', 'X', 'X', 'X', 'X', 'X']),
('y',['o', 'e', 'e', 's', 'g', 'o', 'Y', 'o', 'Y', 'c', 'b', 'D', 'Y', 'e', 'p', 'Y', 'o', 'Y', 'Y', 'Y']),
('z',['Z', 'Z', 'e', 'e', 'z', 'z', 'S', 'S', 'e', 'z', 'z', 'z', 'e', 'Z', 'e', 'z', 'e', 'z', 'S', 'E'])]

class TestCharacterClassifier(unittest.TestCase):
    def test_genHTMLresults(self,
                            reults_array=resultList):
        
        nameHTML = 'charClassifierTest_'+str(TestCharacterClassifier.featExt)+'_'+\
                    str(TestCharacterClassifier.nStrips)+'_'+\
                    ''.join((str(TestCharacterClassifier.nKMClasses)).split('.'))+'_'+\
                    str(TestCharacterClassifier.hmmTopology)+'.html'
        
        #print(nameHTML)
        #db_StallExec(0)
        
        fHTML = open(nameHTML, 'w')

        tScore = Table([
                        ['-']], header_row=(' ','Avg.','character','classification','accuracy',':',' '))
        #tScore.rows.append(TableRow(['SCORE:', '-'], header=True))
        
        #print str(tScore)
        #print '-'*79
    
        tAlpha = Table([
                ['A','-','-','-','-','-','-'],
                ['B','-','-','-','-','-','-'],
                ['C','-','-','-','-','-','-'],
                ['D','-','-','-','-','-','-'],
                ['E','-','-','-','-','-','-'],
                ['F','-','-','-','-','-','-'],
                ['G','-','-','-','-','-','-'],
                ['H','-','-','-','-','-','-'],
                ['I','-','-','-','-','-','-'],
                ['J','-','-','-','-','-','-'],
                ['K','-','-','-','-','-','-'],
                ['L','-','-','-','-','-','-'],
                ['M','-','-','-','-','-','-'],
                ['N','-','-','-','-','-','-'],
                ['O','-','-','-','-','-','-'],
                ['P','-','-','-','-','-','-'],
                ['Q','-','-','-','-','-','-'],
                ['R','-','-','-','-','-','-'],
                ['S','-','-','-','-','-','-'],
                ['T','-','-','-','-','-','-'],
                ['U','-','-','-','-','-','-'],
                ['V','-','-','-','-','-','-'],
                ['W','-','-','-','-','-','-'],
                ['X','-','-','-','-','-','-'],
                ['Y','-','-','-','-','-','-'],
                ['Z','-','-','-','-','-','-'],
                ['a','-','-','-','-','-','-'],
                ['b','-','-','-','-','-','-'],
                ['c','-','-','-','-','-','-'],
                ['d','-','-','-','-','-','-'],
                ['e','-','-','-','-','-','-'],
                ['f','-','-','-','-','-','-'],
                ['g','-','-','-','-','-','-'],
                ['h','-','-','-','-','-','-'],
                ['i','-','-','-','-','-','-'],
                ['j','-','-','-','-','-','-'],
                ['k','-','-','-','-','-','-'],
                ['l','-','-','-','-','-','-'],
                ['m','-','-','-','-','-','-'],
                ['n','-','-','-','-','-','-'],
                ['o','-','-','-','-','-','-'],
                ['p','-','-','-','-','-','-'],
                ['q','-','-','-','-','-','-'],
                ['r','-','-','-','-','-','-'],
                ['s','-','-','-','-','-','-'],
                ['t','-','-','-','-','-','-'],
                ['u','-','-','-','-','-','-'],
                ['v','-','-','-','-','-','-'],
                ['w','-','-','-','-','-','-'],
                ['x','-','-','-','-','-','-'],
                ['y','-','-','-','-','-','-'],
                ['z','-','-','-','-','-','-']
            ], width='100%', header_row=('Alphabet','Rank #1', 'Rank #2','Rank #3', 'Rank #4','Rank #5', 'Others'),
            col_width=('75%'))
        
        total_score = 0
        total_chars = 0
        
        i=0
        for example in reults_array:
            orig_char = example[0]
            l_example = example[1]
            n_elm = len(l_example)
            
            cnt = Counter()
            for letter in l_example:
                cnt[letter]+=1
                
            #print(cnt)
            
            j=0
            
            if(len(cnt) > 5):
                topN = 5
            else:
                topN = len(cnt)
                
            othersCnt=0
            for k,val in cnt.most_common(topN):
                strVal = k+'('+str(int((float(val)/n_elm + 0.00005)*10000.0)/100.0)+'%)'
                if(k == orig_char):
                    tAlpha.rows[i][j+1] = TableCell(strVal, bgcolor='green')
                    total_score += float(val)
                else:
                    tAlpha.rows[i][j+1] = TableCell(strVal)
                othersCnt += float(val)
                j +=1
            
            total_chars += n_elm
            
            if(len(cnt.keys()) >5):
                othersCnt = n_elm - othersCnt 
                strVal = '('+str(int((othersCnt/n_elm + 0.00005)*10000.0)/100.0)+'%)'
                tAlpha.rows[i][j+1] = TableCell(strVal)
            
            i +=1
            db_StallExec(0)
        
        strVal = str(int((total_score/total_chars + 0.00005)*10000.0)/100.0)+'%'
        tScore.rows[0][0] = TableCell(strVal, bgcolor='yellow')
        
        fHTML.write(str(tScore) + '<p>\n')
        fHTML.write(str(tAlpha) + '<p>\n')
        fHTML.close()

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_word_']
    if len(sys.argv) > 1:
        TestCharacterClassifier.hmmTopology = sys.argv.pop()
        TestCharacterClassifier.nKMClasses  = float(sys.argv.pop())
        TestCharacterClassifier.nStrips     = int(sys.argv.pop())
        TestCharacterClassifier.featExt     = sys.argv.pop()
        
    unittest.main()
