# confere se a simulacao em numpy bate com o ZZ feature map do Qiskit
import numpy as np
from qiskit.quantum_info import Statevector
try:
    from qiskit.circuit.library import zz_feature_map as zz
    def circuito(n): return zz(n, reps=2, entanglement="full")
except ImportError:
    from qiskit.circuit.library import ZZFeatureMap
    def circuito(n): return ZZFeatureMap(n, reps=2, entanglement="full")
import quantico

gerador = np.random.default_rng(0)
linhas = []
for n in [2, 3, 4, 5]:
    X = gerador.uniform(-2, 2, size=(6, n))
    mapa = circuito(n)
    est_q = np.array([Statevector(mapa.assign_parameters(x)).data for x in X])
    K_qiskit = np.abs(est_q.conj() @ est_q.T) ** 2
    K_numpy = quantico.kernel_exato(quantico.estados(X), quantico.estados(X))
    erro = np.max(np.abs(K_qiskit - K_numpy))
    linhas.append(f"n={n}: maior diferenca entre os kernels = {erro:.2e}")
    print(linhas[-1])
open("resultados/validacao_qiskit.txt", "w").write("\n".join(linhas) + "\n")
