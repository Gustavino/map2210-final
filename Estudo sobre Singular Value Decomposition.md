# Singular Value Decomposition
## Descrição e aplicação


* Disclaimer (colocar algum aviso em português): na notação BT, o "T" significa que a dada matriz é a transposta.
Por exemplo, FT é a matriz transposta da matriz F.

### Dada (ou dado) uma matriz M, a decomposição SVD entrega, ao final de seu processo, três matrizes, U, Σ e VT, de forma que M = U * Σ * VT.
> Como é feita a decomposição da matriz M:  
    1. É computada uma matriz M\*MT e são calculados os autovetores dessa matriz. Tais autovetores formarão as colunas da matriz U.  
    2. De modo análogo, é computada uma matriz MT\*M e seus autovetores formam as linhas da matriz VT.  
    3. Finalmente, a matriz Σ é formada pelos autovalores de MT\*M.  

* Ilustrações das transformações que cada matriz produzida na fatoração:
    * Procedimento sumarizado:  
    !["SVD transformations"](images/1200px-Singular-Value-Decomposition.svg.png)  
    * Animação com as transformações:  
    !["SVD gif"](images/Singular_value_decomposition.gif)

#### Algoritmo para decomposição SVD em Python puro
```python
import numpy as np

matrix = np.array([[3, 5, 7],
                   [9, 11, 13],
                   [15, 17, 19]])

MMT = matrix @ np.transpose(matrix)
MTM = np.transpose(matrix) @ matrix

mmt_eigvalues, mmt_eigvectors = np.linalg.eig(MMT)
mtm_eigvalues, mtm_eigvectors = np.linalg.eig(MTM)

mmt_eigvalues = np.sqrt(mmt_eigvalues)
mtm_eigvalues = np.sqrt(mtm_eigvalues)

sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
sigma[:matrix.shape[1], :matrix.shape[1]] = np.diag(mtm_eigvalues)

U = mmt_eigvectors
S = sigma
VT = np.transpose(mtm_eigvectors)
```  


#### Cálculo do SVD através do numpy

reformatar esse informações
* INFORMAÇÕES: um valor singular referente a um autovetor é a raiz quadrada do autovalor pertencente a esse mesmo autovetor

* Em uma matriz A, m x n, não-quadrada, a decomposição resulta nas seguintes três matrizes:
    1. U, matriz quadrada, m x m
    2. S (ou Sigma), matriz diagonal, m x n
    3. VT, matriz quadrada, n x n  

!["SVD illustration"](images/svd_matrices.png)

* S, que tem a mesma forma de A, é uma matriz diagonal com os valores singulares de A contidos em sua diagonal principal. 
Portanto, fica evidente que S terá *n* valores singulares e *m-n* linhas de zeros após o último valor singular (ou seja, a partir da linha n+1). 
Assim, as últimas *m-n* colunas de U não afetam o cálculo do produto _U * S_. Dessa forma, existem duas maneiras de calcular a decomposição: 
uma que utiliza as matrizes inteiras e outra que otimiza o cálculo, tirando cômputos desnecessários.

* Na imagem abaixo, o cálculo é otimizado a fim de utilizar somente as primeiras *p* linhas de _S_ e *p* colunas de _U_:  

    !["Optimized SVD"](images/svd_optimized_matrices.jpg)

#### Cálculos utilizando numpy

##### Matriz cheia

```python
import numpy as np


matrix = np.array([[1, 2], [3, 4],
                   [5, 6], [7, 8],
                   [9, 10], [11, 12]])

U, S, VT = np.linalg.svd(matrix)
```  

* Para o método otimizado, o código é quase idêntico, muda-se apenas o parâmetro "full_matrices", que agora recebe "False" como valor.

```python
import numpy as np


matrix = np.array([[1, 2], [3, 4],
                   [5, 6], [7, 8],
                   [9, 10], [11, 12]])

U, S, VT = np.linalg.svd(matrix, full_matrices=False)
```  

#### Cálculos utilizando sklearn

```python

```




### PARA CUSTO COMPUTACIONAL: APLICAR EM DIVERSAS MATRIZES DE ORDEM CRESCENTE COM ENTRADAS REAIS ALEATÓRIAS

### Custo computacional
* No algoritmo para a decomposição SVD de uma matriz A, o custo computacional está em:
    1. Armazenar A, AT, U, Σ e VT em memória;
    2. Calcular dois produtos de matrizes, M\*MT e MT\*M;
        * O produto entre as matrizes, em uma de suas melhores implementações, utilizará o algoritmo de Strassen, que otimiza 
        velocidade em troca de espaço na memória. Uma multiplicação comum entre matrizes leva Θ(n³) operações para ser realizada, 
        enquanto Strassen exige, no pior caso, O(n^2.80).  
    3. E, principalmente, em encontrar dois conjuntos de autovetores e um conjunto de autovalores.  
        * Para este processo, recomenda-se o algoritmo QR com reflexões de Householder, que exige O(9n³) operações. 
        Para efeito de comparações, a versão sem as reflexões demanda O(n⁴) flop.  
        
### Exemplo de aplicação do SVD: compressão de imagens  

```python

```

##### Processo de compressão da imagem, variando 
