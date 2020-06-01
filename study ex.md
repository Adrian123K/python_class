### 잔차제곱합
$$(y-Xw){^T}(y-Xw)$$
$$\sum_{i=1}^N\sum_{j=1}^Na_{ij}x_{i}x_{j}$$
$$ a_{ij} : 가중치$$


### 행렬의 norm
#### 행렬의 모든 요소의 합
```python
np.linalg.norm()
```
$$||A||_{p} = (\sum_{i=1}^N\sum_{j=1}^N|a_{ij}|^p)^{1/p}$$

※ 프로베니우스 Norm 
$$||A||_{F}=||A||=||A||_2=\sqrt{\sum_{i=1}^N\sum_{j=1}^Na_{ij}^2}$$

※ 벡터에서의 Norm
$$ ||x||^2=\sum_{i=1}^Nx_{i}^2=x^{T}x$$

#### Norm 의 성질
1. $||A|| \ge 0 $
2. $||\alpha A|| = |\alpha|\cdot||A||$
3. $||A+B|| \le ||A|| + ||B||$
4. $||AB|| \le ||A||\cdot||B||$

### 행렬의 trace : 대각성분의 총 합(정방행렬에서만!)
```python
np.trace()
```

### 행렬식(determinant) 
$$det$$
$$ |A|$$
```python
np.linalg.det()
```

#### 행렬식의 성질
1. $det(A^T)=det(A)$
2. $det(I)=1$
3. $det(AB)=det(A)det(B)$
4. $det(A^{-1})={1 \over det(A)}$
4. $det(A)\cdot det(A^{-1})=det(I)=1$

#### 역행렬과 전치행렬의 성질
1. $(A^T)^{-1}=(A^{-1})^T$
2. $(AB)^{-1}=B^{-1}A^{-1}$
3. $(AB)^T=B^{T}A^{T}$
4. $(A+B)^{T}=A^T+B^T$
※ 합에 대한 역행렬의 전개식은 존재하지 않음
$(A+B)^{-1} \ne B^{-1}+A^{-1}$


```python
# 행렬의 행렬식 값과 역행렬 구하기 예제
import numpy as np
A=np.array([[1,1,0],[0,1,0],[1,1,1]])
np.linalg.det(A), np.linalg.inv(A)
```

### Boston 집 값 데이터 불러오기


```python
import numpy as np
from sklearn.datasets import load_boston
```


```python
boston=load_boston()
```


```python
X = boston.data
X
```


```python
y = boston.target
y
```


```python
A=X[:4,[0,4,5,6]] #crim:범죄율, nox:공기 오염도, rm:방의 개수, age:오래된 정도
b=y[:4]
A, b
```


```python
w=np.linalg.inv(A)@b # 가중치 계산
w # crim에 반비례, nox에 반비례, rm에 비례, age에 반비례
```

### 최소제곱합을 구하는 방법


```python
# 예시
A=np.array([[1,1,0],[0,1,1],[1,1,1],[1,1,2]])
b=np.array([[2],[2],[3],[4.1]])
A ,b
```


```python
# A의 역행렬
Ainv=np.linalg.inv(A.T @ A) @ A.T
Ainv
```


```python
# 가중치 벡터 구하기
x=Ainv @ b
x
```


```python
A @ x # b과 상당히 유사한 값
```


```python
# lstsq()를 이용하여 한 번에 가중치, 잔차제곱합, 랭크, 특이값(singular value)를 구한다.
x, resid, rank, s = np.linalg.lstsq(A ,b)
x
```


```python
# lstsq()를 사용한 값과 직접 잔차제곱합을 구한 방법
resid, np.linalg.norm(A@x -b) **2 # 잔차 제곱합
```

### Boston 집 값 예제 회귀분석
    1. crim : 범죄율
    2. indus : 비소매 상업지역 면적 비율
    3. nox : 일산화질소 농도
    4. rm : 주택 당 방 수
    5. lstat : 인구 중 하위 계층 비율
    6. B : 인구 중 흑인 비율
    7. ptratio : 학생/교사 비율
    8. zn : 25,000 평방피트를 초과 거주지역 비율
    9. chas : 찰스강의 경계에 위치한 경우는 1, 아니면 0
    10. age : 1940년 이전에 건축된 주택의 비율
    11. rad : 방사형 고속도로까지의 거리
    12. dis : 보스턴 직업 센터 5곳까지의 가중평균거리
    13. tax : 재산세율


```python
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
```


```python
# 가중치 벡터를 구하시오
# Ax=y일 때 x의 값
x, resid, rank, s = np.linalg.lstsq(X, y)
x
```

#### ※ 가중치 벡터의 성분값이 양수이면 정비례관계, 음수이면 반비례관계
