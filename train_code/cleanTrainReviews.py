import sys
import clean_review as cl

def getStemmedDocument(inputFile,outputFile):    
    out = open(outputFile,'w',encoding = 'utf8')
    with open(inputFile,encoding='utf8') as f:
        reviews = f.readlines()
        for review in reviews:
            cleaned_review = cl.parseLine(review)
            print((cleaned_review),file=out)
    out.close()

inputFile = sys.argv[1]
outputFile = sys.argv[2]

getStemmedDocument(inputFile,outputFile)