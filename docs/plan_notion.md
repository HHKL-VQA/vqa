# HHKL + Variational algorithm project

## 1. Project Overview

#### 한 줄 설명

**100-qubit 2D Heisenberg simulation에서 HHKL + variational compilation으로 optimized Trotter보다 낮은 gate count 달성**

#### 프로젝트 세팅

- 10×10 square-lattice Heisenberg model
- (T = 100)
    - QPE context에서 나오는 (T)에 맞춰 조정할 계획
- Error: (0.1, 0.01, 0.001, 0.0001) 중 선택. *error 은 operator norm 기준

#### 예상 산출물

1. HHKL part
    1. HHKL로 어떤 모양으로 lattice를 쪼갤지 결정
    2. 해당 decomposition에서 HHKL error 계산
    3. Variational algorithm에 요구되는 block size / shape, error, gate count, 호출 횟수 계산

-gate를 이기냐 

1. Variational part
    1. Variational circuit architecture / optimization method 결정
    2. 주어진 block size, simulation time, target error에 대한 최소 gate count 산출
2. Baseline 비교 대상
    - 각 target error에 대한 optimized Trotter gate count
3. Integration
    - Total error와 gate count를 계산하여 Trotter와 비교

---

## 2. Research Questions

#### HHKL 관련

1. HHKL decomposition을 어떻게 구성해야 가장 효율적인가?
    - Block shape, 배치 등을 고려해 error와 block 반복 횟수 최소화
2. HHKL theoretical error bound를 어떻게 최적화할 수 있는가?
3. HHKL decomposition 이후의 total error, block 반복 횟수 등을 계산해 variational algorithm에 필요한 정보를 결정할 수 있는가?
4. incorporating higher order product formulas for HHKL(?)

#### Variational Algorithm

1. 작은 block에서 어떤 variational algorithm을 사용해야 target error regime 내에서 gate count를 줄일 수 있는가?
2. Gate count는 얼마나 나오는가?

#### 비교용 Trotter

1. Target error (epsilon)에 대해 gate count가 얼마나 나오는가?
    - Trotter order (p) 및 Trotter step 최적화
2. (선택) Gate depth는 얼마나 나오는가?

#### 그래서 Trotter를 이길 수 있는가?

- HHKL + variational과 optimized Trotter의 gate count 비교
- 추가 방향
    - Gate count 외에 gate depth는 어떻게 되고, 줄일 수 있는가?

---

## 3. Hypotheses / Conjectures

HHKL이 manageable한 block size와 sufficiently large time step을 허용한다.

→ Small-block circuit을 classically optimize할 수 있다.

→ Optimized circuit이 block-level Trotter보다 싸다.

→ Block-level advantage가 HHKL overhead를 상쇄하고도 남는다.

→ 전체 simulation에서 Trotter를 이긴다.

---

## 4. Success Criteria

#### 1. 26년도 단기 목표

대략적인 gate count를 빠르게 산출해, 더 개선하면 Trotter를 이길 수 있는지 판단한다.

#### 2. 장기 목표

FTQC scale에서 우리 방법이 Trotter보다 더 낮은 gate count를 갖는다는 것을 보인다.

---

## 5. Project Decomposition

### 5.1 HHKL

### 5.2 Small-block Circuit Optimization

potential Theoretical directions

### 5.3 Trotter Benchmark

---

## 6 저장용

#### literature

필수

HHKL 알고리즘 https://arxiv.org/abs/1801.03922

선택

lieb robinson bound proof. HHKL 논문에서 증명 읽기 힘들면 이거 보면 되겠습니다.https://arxiv.org/abs/1008.51372. 

local lattice가 되게 특수한 상황이다https://quantum-journal.org/papers/q-2019-07-08-159/3. 

벤치마크 비교 논문https://www.nature.com/articles/s41534-024-00839-44. 

파울리 분해 최적화https://arxiv.org/abs/1403.2975

variational 관련

https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010342

https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033083
