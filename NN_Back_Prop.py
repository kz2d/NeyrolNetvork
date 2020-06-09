from math import exp #it is exp level
class NN:
    def __init__(self,coficent, inputNumber, outputNumber, *intNeyronInLayer):#горизонт- входящие вертикаль- выходящие
        self.coficent=coficent
        self.errors=[[0 for i in range(outputNumber)]]
        self.input=[0 for i in range(inputNumber)]              #self.Wes[номер слоя][номер выходного нейрона][номер входного нейрона]
        self.output=[0 for i in range(outputNumber)]
        self.Wes=[0 for i in range(1+len(intNeyronInLayer))]
        if len(intNeyronInLayer)==0:
            self.Wes[0]=self.ArrayS(inputNumber+1,outputNumber,0.5)
        else:
            for i in intNeyronInLayer:
                self.errors.append([0 for j in range ( i )])
            self.neyrony = [[0 for j in range(i)] for i in intNeyronInLayer]
            self.Wes[0]=self.ArrayS(inputNumber+1, intNeyronInLayer[0],0.5)
            for i in range(1,len(intNeyronInLayer)):
                self.Wes[i]=self.ArrayS(intNeyronInLayer[i-1]+1,intNeyronInLayer[i],0.5)
            self.Wes[-1]=self.ArrayS(intNeyronInLayer[-1]+1,outputNumber,0.5)


    def Work(self, *input):
        self.input=input
        c=input
        for i in range(len(self.Wes)-1):
            for f in range(len(self.neyrony[i])):
                self.neyrony[i][f]=0
                for g in range(len(c)):
                    self.neyrony[i][f]+=self.Wes[i][f][g]*c[g]
                self.neyrony[i][f] += self.Wes[i][f][-1] * 1    #смещение
                self.neyrony[i][f]=self.Sigmoid(self.neyrony[i][f])
            c=self.neyrony[i]

        for i in range(len(self.output)):
            self.output[i]=0
            for h in range(len(c)):
                self.output[i]+=self.Wes[-1][i][h]*c[h]
            self.output[i] += self.Wes[-1][i][-1] * 1 #смещение
            self.output[i]=self.Sigmoid(self.output[i])
        return self.output

    def back_prop(self):
        #for i in range(len(self.))
        if len(self.Wes)>1:
            for c in range ( len ( self.Wes[0] ) ):
                for g in range ( len ( self.Wes[0][c] ) - 1 ):
                    self.Wes[0][c][g] += self.coficent * self.errors[-1][c] * self.neyrony[0][c] * (
                                1 - self.neyrony[0][c]) * self.input[g]
                self.Wes[0][c][-1] += self.coficent * self.errors[-1][c] * self.neyrony[0][c] * (
                            1 - self.neyrony[0][c]) * 1
            for i in range(1,len(self.Wes)-1):
                for c in range(len(self.Wes[i])):
                    for g in range(len(self.Wes[i][c])-1):
                        self.Wes[i][c][g]+=self.coficent*self.errors[-i-1][c]*self.neyrony[i][c]*(1-self.neyrony[i][c])*self.neyrony[i-1][g]
                    self.Wes[i][c][-1]+=self.coficent*self.errors[-i-1][c]*self.neyrony[i][c]*(1-self.neyrony[i][c])*1

            for c in range ( len ( self.Wes[-1] ) ):
                for g in range ( len ( self.Wes[-1][c] ) - 1 ):
                    self.Wes[-1][c][g] += self.coficent * self.errors[0][c] * self.output[c] * (
                                1 - self.output[c]) * self.neyrony[-1][g]
                self.Wes[-1][c][-1] += self.coficent * self.errors[0][c] * self.output[c] * (
                            1 - self.output[c]) * 1
        else:

            for c in range ( len ( self.Wes[0] ) ):
                for g in range ( len ( self.Wes[0][c] ) - 1 ):

                    self.Wes[0][c][g] += self.coficent * self.errors[-1][c] * self.output[c] * (
                            1 - self.output[c]) * self.input[g]
                self.Wes[0][c][-1] += self.coficent * self.errors[-1][c] * self.output[c] * (
                        1 - self.output[c]) * 1

    def error(self, *label):
        for i in range(len(self.errors[0])):
            self.errors[0][i]=-label[i]+self.output[i]
        for i in range(1,len(self.errors)):
            for j in range(len(self.errors[i])):
                self.errors[i][j]=0
                for g in range(len(self.errors[i-1])):
                    self.errors[i][j]+=self.Wes[-i][g][j]*self.errors[i-1][g]

    def lern(self,input:list, output:list):
        self.Work(*input)
        self.error(*output)
        self.back_prop()

    def Sigmoid(self, x):
        return 1/(1+exp(x))

    def show(self):
        j=0
        for i in self.Wes:
            print(j)
            for c in i:
                print(c)
            j=+1

    def showResult(self):
        j = 0
        for i in self.neyrony:
            print ( j )
            for c in i:
                print ( c )
            j = +1

    def ArrayS(self, int1, int2, basic):
        return [[basic for j in range(int1)] for i in range(int2)]

# v= NN(0.5,2,3,2,2)
# #v.Wes=[[[0.5,0.1,0.1],[0.2,0.2,0.3]],[[0.5,0.1,0.1],[0.2,0.2,0.3]]]
# print(v.errors)
# print(v.Work(1,5))
# v.error(1,1,1)
# print(v.errors)
# for i in range(1000):
#     v.Work(1,5,5)
#     v.error(1,1,1)
#     v.back_prop()
#     v.Work ( 1 , 3 ,7)
#     v.error ( 1 , 1 ,1)
#     v.back_prop ()
#     print(v.errors)
# print(v.Work(1,5,5))