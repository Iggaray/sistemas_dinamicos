#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:12:57 2021

@author: nacho
"""
import numpy as np
import sympy as sy
from sympy.solvers.solveset import nonlinsolve

class SistemaDinamico(object):
    '''
    x es una lista de simbolos sympy
    f es una lista de expresiones sympy con las derivadas de cada variable
    '''
    def __init__(self, x, f):
        self.x = x
        self.f = f
        #genero el jacobiano
        self.J = []
        for fun in f:
            for var in x:
                self.J.append(sy.diff(fun, var))
        self.J = np.array(self.J).reshape((len(self.x), len(self.f)))
        self.J = sy.Matrix(self.J)
    
    def puntos_fijos(self):
        ''' encuentra todos los puntos fijos asociados
        los guarda internamente pero tambien los devuelve'''
        self.p_fijos = nonlinsolve(self.f, self.x)
        return self.p_fijos
    
    def estabilidad(self):
        ''' crea una lista de dicts de autovalores para cada punto fijo'''
        self.avals=[]
        for p in self.p_fijos:
            J_eval = self.J
            for i in range(len(self.x)):
                J_eval=J_eval.subs(self.x[i], p[i])
            self.avals.append(J_eval.eigenvals())
    
    def evol():
        pass #pendiente para graficar evolucion
    
    def print_avals(self):
        '''imprime los autovalores de cada punto fijo'''
        for p in self.avals:
            for key in p.keys():
                print(sy.latex(key.simplify()))
            print("\n")
    
    def print_pfijos(self):
        for p in self.p_fijos:
            print(sy.latex(p.simplify()))
            print("\n")
    
    def streamplot(self, x_range, y_range):
        '''Solo sistemas 2D
        x, y: 1D numpy arrays con el rango de interes en cada variable
        Internamente se hace un meshgrid'''
        if len(self.f)!=2 or len(self.x)!=2:
            print("No es sistema 2D o hay parametros no definidos")
            return 0
        
        fx = sy.lambdify(self.x, self.f[0])
        fy = sy.lambdify(self.x, self.f[1])
        xx, yy = np.meshgrid(x_range, y_range)
        plt.streamplot(xx, yy, fx(xx, yy), fy(xx, yy))
