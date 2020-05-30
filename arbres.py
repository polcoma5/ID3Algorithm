'''
Pol Coma Barbara ( 1426607 )
Xavi Coret Mayoral ( 1423586 )
Marti Tuneu Font ( 1425069 )
'''
from random import *
from collections import OrderedDict
import numpy as np
import math

###############################################################################################
###		Funcions de tractament de fitxers													###
###############################################################################################
class Reader:
	def __init__(self,database,info,substitutionType=0,boolId=0):
		self.__database=open(database, 'r')
		self.__dataInfo=open(info,'r')
		self.__readedData=[]
		self.__dataPossibilities=OrderedDict()
		self.__attributes=OrderedDict
		self.__attributesDomains=[]
		self.__totalExamples=0
		self.canUse=[]
		self.realValues=[]
		self.__cleanedData=[]
		self.__boolId=boolId
		self.__chargeData()
		self.__chargeDataInfo()
		self.__cleanData(substitutionType)
		self.__definitiveCleaning()
		self.__dataInfo.close()
		self.__database.close()
###############################################################################################
	def __chargeData(self):
		for data in self.__database:
			eachReadedData=self.__cleanReadedData(data)
			self.__readedData.append(eachReadedData)
###############################################################################################
	def __cleanReadedData(self,data):
		data.replace(' ', '')
		cleanedData=data.split(',')
		cleanedData[-1]=cleanedData[-1][0:-1]
		if self.__boolId:
			cleanedData=cleanedData[1:]
		return cleanedData
###############################################################################################
	def __chargeDataInfo(self):
		self.eachcharacteristicInfo=[]
		for data in self.__dataInfo:
			eachReadedData=self.__cleanReadedData(data)
			self.eachcharacteristicInfo.append(eachReadedData)
		self.__attributes=self.eachcharacteristicInfo[0]
		self.__attributes[0]=self.__attributes[0][1:]
		self.__attributesDomains=self.eachcharacteristicInfo[1]
		self.__generateTypeOfData()
###############################################################################################
	def __generateTypeOfData(self):
		for attribute,domain in zip(self.__attributes,self.__attributesDomains):
			if domain == 'continuous':
				self.realValues.append(attribute)
			else:
				self.canUse.append(attribute)
		self.canUse=self.canUse[:-1]
###############################################################################################
	def __cleanData(self,substitutionType=0):
		if substitutionType==0:
			self.__deleteNoneValues()
		elif substitutionType==1:
			self.__substituteNonesForMean()
		else:
			self.__substituteNonesForMode()
		self.__totalExamples=len(self.__readedData)
		return True
###############################################################################################
	def __deleteNoneValues(self):
		self.__readedData=[x for x in self.__readedData if '?' not in x]
###############################################################################################				
	def __substituteNonesForMean(self):
		dataForIter = list(zip(*self.__readedData))
		nValues=self.__totalExamples
		for indexCharacteristic,eachCaracteristic in enumerate(dataForIter):
			mean=self.__calculateMean(eachCharacteristic,nValues)
			self.__substituteForMean(eachCharacteristic,mean)
		self.__readedData = list(zip(*dataForIter))
###############################################################################################
	def __calculateMean(self,nValues):
		totalSum=0
		for eachParticularCharacteristic in self.__readedData:
			if eachParticularCharacteristic != '?':
				totalSum+=eachParticularCharacteristic
				nValues-=1
			mean=totalSum/nValues
		return mean
###############################################################################################
	def __substituteNonesForMode(self):	
		dataForIter=list(zip(*self.__readedData))[:]
		for indexCharacteristic,eachCharacteristic in enumerate(dataForIter):
			mode=self.__calculateMode(eachCharacteristic)
			eachCharacteristic=list(eachCharacteristic)
			self.__substituteForMode(eachCharacteristic,mode)
		self.__readedData = zip(*dataForIter)[:]
###############################################################################################
	def __calculateMode(self,eachCharacteristic):
		common_values={}
		for entry in eachCharacteristic:
			if common_values.has_key(entry):
				common_values[entry] += 1
			else:
				common_values[entry]  = 1
		return max(common_values, key=common_values.get) 
###############################################################################################
	def __substituteForMode(self,eachCharacteristic,mode):
		for data in eachCharacteristic:
			if data=='?':
				eachCharacteristic[eachCharacteristic.index(data)] = mode
###############################################################################################
	def crossValidation(self,nPartitions=10):
		if type(nPartitions) != int:
			return [-1],[-1]
		partitionSize=len(self.__readedData)/nPartitions
		index=0
		partitions=[]
		for i in range(nPartitions-1):
			partitions.append(self.__readedData[index:index+partitionSize])
			index+=partitionSize
		partitions.append(self.__readedData[index:])
		return partitions
###############################################################################################
	def leaveOneOut(self):		
		trainSet=self.__readedData[:]
		testValue=randint(0, self.__totalExamples-2)
		testSet=trainSet[testValue][:]
		del trainSet[testValue]
		return trainSet,testSet
###############################################################################################
	def __definitiveCleaning(self):
		for attribute in self.canUse:
			self.__attributes.index(attribute)

###############################################################################################
###		Classe Node 																		###
###############################################################################################
def evaluation(Tree,testSet,method,dataSetName,attributesUsed):
	if dataSetName=='breastCancer.data':
		TRUE='4'
		FALSE='2'
	else:
		CASE1='u'
		CASE2='g'
		CASE3='m'
		CASE4='p'		
	efficiency={'TP':0,'TN':0,'FP':0,'FN':0}
	precision=0
	recall=0
	datasetBinomial=1
	if dataSetName=='mushrooms.data':
		datasetBinomial=0
	if method==0:#leave one out
		comprobation=Tree
		while(not comprobation.isLeave):
			label=comprobation.label
			correspondentIndex=attributesUsed.keys().index(label)
			value=testSet[correspondentIndex]
			if not comprobation.childs.has_key(value):
				efficiency['TN']+=1
				break
			comprobation=comprobation.childs[value]
		if datasetBinomial==True:
			if comprobation.finalClassification == testSet[-1] :
				if testSet[-1]==TRUE:
					efficiency['TP']+=1
				else:
					efficiency['TN']+=1
			else:
				if testSet[-1]==TRUE:
					efficiency['FP']+=1
				else:
					efficiency['FN']+=1
		else:
			if comprobation.finalClassification == testSet[-1] :
				efficiency['TP']+=1
			else:
				efficiency['TN']+=1	
	else:#cross validation
		for diffTest in testSet:
			comprobation=Tree
			counter=0
			while(not comprobation.isLeave):
				label=comprobation.label 
				if label !='':
					correspondentIndex=attributesUsed.keys().index(label)
					value=diffTest[correspondentIndex]
				if comprobation.childs.has_key(value):
					comprobation=comprobation.childs[value]
				else:
					break
			if datasetBinomial==True:

				if comprobation.finalClassification == diffTest[-1]:
					if diffTest[-1]==TRUE:
						efficiency['TP']+=1
					else:
						efficiency['TN']+=1		
				else:
					if diffTest[-1]==TRUE:
						efficiency['FP']+=1
					else:
						efficiency['FN']+=1
			else:
				if comprobation.finalClassification == testSet[-1] :
					efficiency['TP']+=1		
				else:
					efficiency['TN']+=1
	accuracy=(1.0*efficiency['TP']+efficiency['TN'])/(efficiency['TP']+efficiency['TN']+efficiency['FP']+efficiency['FN'])
	if efficiency['TP']>0 or efficiency['FP']>0:
		precision=float(efficiency['TP'])/(efficiency['TP']+efficiency['FP'])
	if efficiency['TP']>0 or efficiency['FN']>0:
		recall=float(efficiency['TP'])/(efficiency['TP'] + efficiency['FN'])
	if efficiency['FP']>0 or efficiency['TN']>0:
		specificity=float(efficiency['TN'])/(efficiency['TN'] + efficiency['FP'])
	if precision>0 or recall>0:
		f_measure= (2.0*precision*recall)/(precision+recall)
	print 'Accuracy: '+ str(accuracy)

###############################################################################################
###		Classe Node 																		###
###############################################################################################
class Node:
    def __init__(self,prof,parent,dataNode,actualEntropy,connectionName,isLeave=False,root=None,finalClassification='No'):
        self.parent=parent
        self.childs={}
        self.dataNode=dataNode
        self.actualEntropy=actualEntropy
        self.isLeave=False
        self.connectionName=connectionName
        self.prof=prof
        self.root=root
        self.label=None
        self.finalClassification=finalClassification
        self.setLevel()

    def setLevel(self):
        if self.parent == None:
            self.level=0
        else:
            self.level = self.parent.level+1
    def setRoot(self):
        self.root=self.parent.root
    def autoParent(self):
        self.parent=self

###############################################################################################
###		Funcions per al tractament i construccio de arbres									###
###############################################################################################
def splitCriterionID3(allData,realAttributes,validAttributes,actualEntropy):
    maxGain=0
    maxGainPossiblitities=[]
    attributeWithMaxGain=''
    newEntropy=0
    bestEntropy=0
    listIndexValidAttributes=genValidAttributeListIndexes(validAttributes)
    for indexAttribute in listIndexValidAttributes:
        attributeGain,possibilities,newEntropy=calcGain(allData,indexAttribute,actualEntropy)
        if attributeGain > maxGain:
            maxGain=attributeGain
            attributeWithMaxGain=extractValidAttributeName(indexAttribute,validAttributes)
            maxGainPossiblitities=possibilities
            bestEntropy=newEntropy
    if validAttributes.has_key(attributeWithMaxGain):
        validAttributes[attributeWithMaxGain]=0
    return attributeWithMaxGain,maxGainPossiblitities,bestEntropy,validAttributes

###############################################################################################
def splitCriterionC45(allData,realAttributes,validAttributes,actualEntropy):
    maxGainRatio=0
    maxGainPossiblitities=[]
    attributeWithMaxGain=''
    newEntropy=0
    bestEntropy=0
    listIndexValidAttributes=genValidAttributeListIndexes(validAttributes)
    for indexAttribute in listIndexValidAttributes:
        attributeGainRatio=0
        attributeGain,possibilities,newEntropy=calcGain(allData,indexAttribute,actualEntropy) 
        valFreq={}

        for entry in allData:
            if (valFreq.has_key(entry[indexAttribute])):
                valFreq[entry[indexAttribute]] += 1.0
            else:
                valFreq[entry[indexAttribute]]  = 1.0

        totalValues=len(allData)
        attributeSplitInfo=calcSplitInfo(valFreq,totalValues)
        if attributeSplitInfo!=0:
            attributeGainRatio=attributeGain/attributeSplitInfo

        if attributeGainRatio > maxGainRatio:
            maxGainRatio=attributeGainRatio
            maxGain=attributeGain
            attributeWithMaxGain=extractValidAttributeName(indexAttribute,validAttributes)
            maxGainPossiblitities=possibilities
            bestEntropy=newEntropy

    if validAttributes.has_key(attributeWithMaxGain):
        validAttributes[attributeWithMaxGain]=0
    return attributeWithMaxGain,maxGainPossiblitities,bestEntropy

###############################################################################################
def extractValidAttributeName(indexAttribute,validAttributes):
    AttributeList=validAttributes.keys()
    return AttributeList[indexAttribute]
###############################################################################################
def genValidAttributeListIndexes(validAttributes):
    return [index for index,attribute in enumerate(validAttributes.keys()) if validAttributes[attribute] == 1]
###############################################################################################
def calcSplitInfo(diffValues,totalValues):
    totalSum=0
    for sv in diffValues.keys():
        totalSum+=(-1.0*int(diffValues[sv])/totalValues)* math.log(1.0*int(diffValues[sv])/totalValues,2)
    return totalSum
###############################################################################################
def calcGain(data,attributeIndex,actualEntropy):
    subsetEntropy=0
    valFreq={}
    for entry in data:
        if (valFreq.has_key(entry[attributeIndex])):
            valFreq[entry[attributeIndex]] += 1.0
        else:
            valFreq[entry[attributeIndex]]  = 1.0
    for val in valFreq.keys():
        valProb        = 1.0*valFreq[val] / sum(valFreq.values())
        dataSubset     = [entry for entry in data if entry[attributeIndex] == val]
        subsetEntropy += valProb * calcEntropy(dataSubset)
    return (calcEntropy(data) - subsetEntropy),valFreq.keys(),subsetEntropy

###############################################################################################
def calcEntropy(data):
    valFreq={}
    for entry in data:
        if (valFreq.has_key(entry[-1])):
            valFreq[entry[-1]] += 1.0
        else:
            valFreq[entry[-1]]  = 1.0
    dataEntropy=0
    for freq in valFreq.values():
        dataEntropy += (-freq/len(data)) * math.log(1.0*freq/len(data), 2) 
    return dataEntropy

###############################################################################################
def StoppingCriterion(data):
    if(calcEntropy(data)<0.01):
        return True
    return False
###############################################################################################
def assignBestLabel(data):
    return data[0][-1]
###############################################################################################
def dataSeparatedByParameter(data,specificValueOfAttribute,attributeName,attributeList,attributeListContinuous):
    attributes=attributeList.keys()
    indexOfAttribute=attributes.index(attributeName)

    dataExtractedFromAttribute=[]
    for data in data:
        if data[indexOfAttribute]==specificValueOfAttribute:
            dataExtractedFromAttribute.append(data)

    return dataExtractedFromAttribute
###############################################################################################
def treeGenerationID3(data,attributeList,attributeContinuousList,parent=None,entropy=1,connectionName='root',prof=0,root=None):
	Tree=Node(prof,parent,data,entropy,connectionName,False,root,'No')
	if prof==0:
		Tree.autoParent()
		Tree.setRoot()
	if(StoppingCriterion(Tree.dataNode)):
		Tree.isLeave=True
		bestLabel=assignBestLabel(data)
		Tree.finalClassification=bestLabel
		Tree.label=bestLabel
	else:
		Tree.label,listOfCharacteristics,newEntropy,attributeList=splitCriterionID3(Tree.dataNode,attributeContinuousList,attributeList,entropy)
		for eachPossibleValue in listOfCharacteristics:
			Tree.dataNode=dataSeparatedByParameter(data,eachPossibleValue,Tree.label,attributeList,attributeContinuousList)
			subTree=treeGenerationID3(Tree.dataNode,attributeList,attributeContinuousList,Tree,newEntropy,eachPossibleValue,prof+1,Tree.root)
			Tree.childs[eachPossibleValue]=subTree
	return Tree
###############################################################################################
def treeGenerationC45(data,attributeList,attributeContinuousList,parent=None,entropy=1,connectionName='root',prof=0,root=None):
	Tree=Node(prof,parent,data,entropy,connectionName,False,root,'No')
	if prof==0:
		Tree.autoParent()
		Tree.setRoot()
	if(StoppingCriterion(Tree.dataNode)):
		Tree.isLeave=True
		bestLabel=assignBestLabel(data)
		Tree.finalClassification=bestLabel
		Tree.label=bestLabel
	else:
		Tree.label,listOfCharacteristics,newEntropy=splitCriterionC45(Tree.dataNode,attributeContinuousList,attributeList,entropy)
		for eachPossibleValue in listOfCharacteristics:
			Tree.dataNode=dataSeparatedByParameter(data,eachPossibleValue,Tree.label,attributeList,attributeContinuousList)
			subTree=treeGenerationC45(Tree.dataNode,attributeList,attributeContinuousList,Tree,newEntropy,eachPossibleValue,prof+1,Tree.root)
			Tree.childs[eachPossibleValue]=subTree
	
	return Tree
###############################################################################################
def printTree(Tree):
	listNodesToVisit=[]
	FinalList=[]
	eachcharacteristics = reader.eachcharacteristicInfo[0]
	listNodesToVisit.extend(Tree.childs.values())
	Tree.connectionName='ROOT'
	profFinal=findLastNode(Tree).prof
	print str(Tree.connectionName)+'  '+ Tree.label
	while(listNodesToVisit):
		actualNode=listNodesToVisit[0]
		if actualNode.childs:
			for child in actualNode.childs.values():
				listNodesToVisit.insert(0,child)
		if actualNode.label != '':
			tabs=''
			for i in range(actualNode.prof):
				tabs=tabs+'  '
			flag = True
			try:
				tag = int(actualNode.label)
			except:
				flag=False
				tag = actualNode.label
			if flag:
				print tabs+str(actualNode.connectionName)+':'+eachcharacteristics[int(actualNode.connectionName)-1]
			else:
				print tabs+str(actualNode.connectionName)+':'+tag
			
		del listNodesToVisit[listNodesToVisit.index(actualNode)]
###############################################################################################
def findLastNode(Tree):
	actualNode=Tree
	while(not actualNode.isLeave):
		differentNodesList=actualNode.childs.values()
		if not differentNodesList:
			break
		actualNode=differentNodesList[-1]
	actualNode.isLastNode=True
	return actualNode
###############################################################################################
def generateSets(partitions):
	testSet=[]
	trainSet=[]
	for p in partitions[:-1]:
		trainSet.extend(p)
	testSet=partitions[-1]
	return testSet,trainSet
###############################################################################################
def initAttributes(AttributeList):
	return {attribute: 1 for attribute in AttributeList}

###############################################################################################
###		Main																				###
###############################################################################################
if __name__=='__main__':
	substitutionType=0
	nPartitions=5	
	boolId=1
	reader=Reader('breastCancer.data','breastCancerInfo.data',substitutionType,boolId)
	print 'Base de Dades: breastCancer'
	trainSet,testSet=reader.leaveOneOut()
	partitions=reader.crossValidation(nPartitions)
	testSet,trainSet=generateSets(partitions)
	if testSet[0] ==[-1]:
		print "Error en el nombre de particions"
	else:
		boolMultiValor=initAttributes(reader.canUse)
		attributesContinuousUsed=initAttributes(reader.realValues)
		trainSet45=trainSet[:]
		ID3=treeGenerationID3(trainSet,boolMultiValor,attributesContinuousUsed)
		boolMultiValor=initAttributes(reader.canUse)
		C45=treeGenerationC45(trainSet45,boolMultiValor,attributesContinuousUsed)
		print 'Algorisme ID3:'
		printTree(ID3)
		print ''
		print 'Algorisme C4.5:'
		printTree(C45)
		method=1#cross validation
		evaluation(C45,testSet,method,'breastCancer.data',boolMultiValor)