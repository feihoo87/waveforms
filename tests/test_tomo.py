import numpy as np
from waveforms.quantum.math import randomDensity
from waveforms.quantum.tomo import (U_to_chi, applyOp, dagger, qpt, qptBases,
                                    qptInitList, qst_mle, qstOpList,
                                    tensorMatrix)


def test_qst():
    # QST
    N = 2  # 两个比特
    rho = randomDensity(2**N)

    # 模拟实验采集数据
    P = []
    for op in qstOpList(N):
        # 先做 op 变换，再测量 population
        z = np.diag(tensorMatrix(op) @ rho @ dagger(tensorMatrix(op)))
        P.append(z)

    rho_exp = qst_mle(P)
    assert np.allclose(rho, rho_exp, atol=0.01)


def test_qpt():
    # QPT
    riList = []
    rfList = []

    N = 2
    testGate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                         [0, 0, 0, -1]])

    for U in qptInitList(N):
        # 分别制备不同的初态
        ri = applyOp(U)
        # 做完操作后测末态的 QST
        rf = testGate @ ri @ dagger(testGate)
        riList.append(ri)
        rfList.append(rf)

    # 计算 chi 超算符
    chi = qpt(riList, rfList)

    def checkChi(chi, U):
        rho_0 = randomDensity(U.shape[0])
        rho_F = np.zeros_like(U, dtype=complex)
        for m, Pm in enumerate(qptBases(N)):
            for n, Pn in enumerate(qptBases(N)):
                rho_F += chi[m, n] * (
                    tensorMatrix(Pm) @ rho_0 @ dagger(tensorMatrix(Pn)))
        return np.allclose(rho_F, U @ rho_0 @ dagger(U))

    assert checkChi(chi, testGate)
    assert np.allclose(chi, U_to_chi(testGate))
