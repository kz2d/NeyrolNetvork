import numpy as np
from math import exp
import plotly.express as px
class NN:
    def __init__(self, coficent, inputN, outpN, hideN):
        self.cof=coficent
        self.wes0=np.random.rand(hideN, inputN)-0.5
        self.wes1=np.random.rand(outpN, hideN)-0.5
        self.inp=np.zeros(inputN)
        self.hide=np.zeros(hideN)
        self.outp=np.zeros(outpN)
        print(self.inp)

    def activ(self, x):
        for y in range(len(x)):
            x[y, 0] = 1/(1+exp(-x[y, 0]))

    def Work(self, inpList):
        self.inp=np.array(inpList, ndmin = 2).T
        self.hide=np.dot(self.wes0, self.inp)
        print(self.hide)
        self.activ(self.hide)
        print(self.hide)
        self.outp=np.dot(self.wes1, self.hide)
        self.activ(self.outp)
        return self.outp

    def train(self, inpList, otpList):
        self.Work(inpList)
        target = np.array(otpList, ndmin = 2).T
        print(target)
        print(self.outp)
        self.errors=target-self.outp
        print(self.errors)
        hide_errors=np.dot(self.wes1.T, self.errors)
        self.wes1+=self.cof*np.dot(self.errors*self.outp*(1-self.outp),np.transpose(self.hide))
        self.wes0+=self.cof*np.dot(hide_errors*self.hide*(1-self.hide),np.transpose(self.inp))
