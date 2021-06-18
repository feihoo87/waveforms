import numpy as np
from waveforms.quantum.math import randomDensity, randomUnitary
from waveforms.quantum.tomo import (U_to_chi, applyChi, applyOp, dagger,
                                    pauli_basis, qpt, qptBases, qptInitList,
                                    qst_mle, qstOpList, tensorMatrix)


def applyChiDef(chi, rho, basis=pauli_basis):
    ret = np.zeros_like(rho, dtype=complex)
    N = int(round(np.log2(ret.shape[0])))
    for i, m in enumerate(qptBases(N, gates=basis)):
        for j, n in enumerate(qptBases(N, gates=basis)):
            ret += chi[i, j] * tensorMatrix(m) @ rho @ dagger(tensorMatrix(n))
    return ret


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
    testGate = randomUnitary(4)

    for U in qptInitList(N):
        # 分别制备不同的初态
        ri = applyOp(U)
        # 做完操作后测末态的 QST
        rf = testGate @ ri @ dagger(testGate)
        riList.append(ri)
        rfList.append(rf)

    # 计算 chi 超算符
    chi = qpt(riList, rfList)

    assert np.allclose(chi, U_to_chi(testGate))

    rho_0 = randomDensity(testGate.shape[0])
    assert np.allclose(testGate @ rho_0 @ dagger(testGate),
                       applyChi(chi, rho_0))

    def checkChi(chi, U):
        rho_0 = randomDensity(U.shape[0])
        rho_F = applyChiDef(chi, rho_0)
        return np.allclose(rho_F, U @ rho_0 @ dagger(U))

    assert checkChi(chi, testGate)
    assert np.allclose(chi, U_to_chi(testGate))
    assert np.allclose(chi, qpt(rfList))
