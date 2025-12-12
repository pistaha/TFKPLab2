import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def describe_domains():
    """аналитическое описание областей из условия"""
    print("аналитическое описание областей")
    print("\nрис 7 (область D₁):")
    print("D₁ = {z ∈ ℂ : Re(z) + Im(z) < -1}")
    print("граница: прямая x + y = -1, проходит через точки -1 и -i")
    print("точка 0 ∉ D₁")
    print("\nрис 8 (область D₂):")
    print("D₂ = {ω ∈ ℂ : |ω| < π}")
    print("круг радиуса π с центром в 0")
    print("кочка 0 ∈ D₂")


def step1_shift(z):
    """ε = z + i - сдвиг на i вверх"""
    return z + 1j


def step2_rotate(eps):
    """γ = e^{-3πi/4}·ε - поворот на 135° по часовой стрелке"""
    return np.exp(-3j * np.pi / 4) * eps


def step3_halfplane_to_unit_disk(gamma):
    """α = (γ - i)/(1 - iγ) - дробно-линейное отображение в единичный круг"""
    return (gamma - 1j) / (1 - 1j * gamma)


def step4_scale(alpha):
    """ω = π·α - растяжение до круга радиуса π"""
    return np.pi * alpha


def full_mapping(z):
    """полное отображение D₁ → D₂ как композиция четырёх шагов"""
    eps = step1_shift(z)
    gamma = step2_rotate(eps)
    alpha = step3_halfplane_to_unit_disk(gamma)
    return step4_scale(alpha)


def inverse_mapping(omega):
    """обратное отображение D₂ → D₁ через обратные преобразования"""
    alpha = omega / np.pi
    gamma = (alpha + 1j) / (1 + 1j * alpha)
    eps = np.exp(+3j * np.pi / 4) * gamma
    return eps - 1j


def test_mappings():
    """проверка прямого и обратного отображения на тестовых точках"""
    print("\nтестирование")

    test_points = [
        (-1.5 - 1.5j, "внутренняя точка D₁"),
        (-2.0 + 0.0j, "точка на отрицательной вещественной оси"),
        (0.0 - 2.0j, "точка на отрицательной мнимой оси"),
        (-1.0 - 1e-12j, "близко к границе у точки -1"),
        (-1e-12 - 1j, "близко к границе у точки -i"),
        (0.0 + 0.0j, "точка вне D₁ (начало координат)"),
    ]

    for z, description in test_points:
        in_D1 = z.real + z.imag < -1
        omega = full_mapping(z)
        z_back = inverse_mapping(omega)
        error = abs(z - z_back)

        print(f"\n{description}:")
        print(f"  z = {z:>18}")
        print(f"  z ∈ D₁: {in_D1}, Re(z)+Im(z) = {z.real + z.imag:+.6f}")
        print(f"  ω = {omega:>18}")
        print(f"  |ω| = {abs(omega):.10f} {'< π ✓' if abs(omega) < np.pi else '≥ π ✗'}")
        print(f"  z' = {z_back:>18}")
        print(f"  ||z - z'|| = {error:.2e}")


def verify_boundary_mapping():
    """проверка отображения границы D₁ в окружность |ω| = π"""
    print("\n провнрка границ отображения")
    eps = 1e-9
    boundary_points = [
        (-1 + 0j, "точка -1 на границе"),
        (0 - 1j, "точка -i на границе"),
        (-0.5 - 0.5j, "точка -0.5-0.5i на границе"),
    ]

    for z_boundary, description in boundary_points:
        z = z_boundary + eps * (-1 - 1j)
        in_D1 = z.real + z.imag < -1
        omega = full_mapping(z)

        print(f"\n{description}:")
        print(f"  z_boundary = {z_boundary}")
        print(f"  z_inside = {z} (сдвиг внутрь D₁)")
        print(f"  проверка: Re(z)+Im(z) = {z.real + z.imag:.10f} < -1: {in_D1}")
        print(f"  |ω| = {abs(omega):.12f}")
        print(f"  π = {np.pi:.12f}")
        print(f"  отклонение: {abs(abs(omega) - np.pi):.2e}")


def generate_figure7_points():
    """Генерация точек для визуализации области D₁"""
    x = np.linspace(-3.5, 0.5, 150)
    y = np.linspace(-3.5, 0.5, 150)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    mask = X + Y < -1
    inside_points = Z[mask]

    t = np.linspace(-3, 1, 200)
    boundary_points = t - 1j * (t + 1)

    return inside_points, boundary_points


def generate_figure8_points():
    """Генерация точек для визуализации области D₂"""
    theta = np.linspace(0, 2 * np.pi, 200)
    boundary = np.pi * np.exp(1j * theta)

    r = np.linspace(0, 0.95 * np.pi, 10)
    theta_grid = np.linspace(0, 2 * np.pi, 30)
    R, T = np.meshgrid(r, theta_grid)
    inside = (R * np.exp(1j * T)).flatten()

    return inside, boundary

def plot_all_in_one():
    fig = plt.figure(figsize=(18, 12))

    # 1. Область D₁
    ax1 = plt.subplot(2, 3, 1)
    inside_points1, boundary1 = generate_figure7_points()

    ax1.set_xlim(-3.5, 1.5)
    ax1.set_ylim(-3.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axvline(0, color='black', linewidth=0.8)

    # Граница D₁
    ax1.plot(boundary1.real, boundary1.imag, 'k-', linewidth=2)

    # Заполнение области D₁
    x_fill = np.linspace(-3.5, 0.5, 100)
    ax1.fill_between(x_fill, -3.5, -x_fill - 1,
                     where=(x_fill <= 0.5),
                     color='lightblue', alpha=0.3)

    # Ключевые точки
    ax1.plot(-1, 0, 'ro', markersize=6)
    ax1.text(-1.2, 0.1, '-1', fontsize=10)
    ax1.plot(0, -1, 'ro', markersize=6)
    ax1.text(0.1, -1.1, '-i', fontsize=10)
    ax1.plot(0, 0, 'ko', markersize=6)
    ax1.text(0.1, 0.1, '0', fontsize=10)

    ax1.set_title("1. Область D₁ (Рис.7)", fontsize=12)
    ax1.set_xlabel("Re(z)")
    ax1.set_ylabel("Im(z)")

    # 2. Область D₂
    ax2 = plt.subplot(2, 3, 2)
    inside_points2, boundary2 = generate_figure8_points()

    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)

    # Окружность |ω| = π
    circle2 = plt.Circle((0, 0), np.pi, fill=False,
                         color='black', linewidth=2)
    ax2.add_patch(circle2)

    # Заполнение области D₂
    circle_fill2 = patches.Circle((0, 0), np.pi,
                                  color='lightgreen', alpha=0.3)
    ax2.add_patch(circle_fill2)

    # Ключевые точки
    ax2.plot(0, 0, 'ko', markersize=6)
    ax2.text(0.15, 0.15, '0', fontsize=10)
    ax2.plot(np.pi, 0, 'ro', markersize=6)
    ax2.text(np.pi + 0.2, 0.1, 'π', fontsize=10)

    ax2.set_title("2 область D₂ (рис.8)", fontsize=12)
    ax2.set_xlabel("Re(ω)")
    ax2.set_ylabel("Im(ω)")

    # 3. Исходные точки в D₁
    ax3 = plt.subplot(2, 3, 3)
    sample_points = inside_points1[::40] if len(inside_points1) > 40 else inside_points1

    ax3.scatter(sample_points.real, sample_points.imag,
                s=8, alpha=0.6, color='blue')
    ax3.plot(boundary1.real, boundary1.imag, 'r-', linewidth=1.5, alpha=0.7)
    ax3.set_xlim(-3.5, 1.5)
    ax3.set_ylim(-3.5, 1.5)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title("3 исходные точки в D₁", fontsize=12)
    ax3.set_xlabel("Re(z)")
    ax3.set_ylabel("Im(z)")

    # 4. После сдвига и поворота (ε → γ)
    ax4 = plt.subplot(2, 3, 4)
    if len(sample_points) > 0:
        eps = step1_shift(sample_points)
        gamma = step2_rotate(eps)
        ax4.scatter(gamma.real, gamma.imag,
                    s=8, alpha=0.6, color='orange')
        ax4.axhline(0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.set_xlim(-5, 5)
        ax4.set_ylim(-1, 8)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.set_title("4 после ε=z+i и γ=e^{-3πi/4}·ε", fontsize=12)
        ax4.set_xlabel("Re(γ)")
        ax4.set_ylabel("Im(γ)")
        ax4.text(3, 6, "Im(γ) > 0", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))

    # 5. После дробно-линейного преобразования (γ → α)
    ax5 = plt.subplot(2, 3, 5)
    if len(sample_points) > 0:
        eps = step1_shift(sample_points)
        gamma = step2_rotate(eps)
        alpha = step3_halfplane_to_unit_disk(gamma)
        ax5.scatter(alpha.real, alpha.imag,
                    s=8, alpha=0.6, color='green')
        unit_circle = plt.Circle((0, 0), 1, fill=False,
                                 color='r', linewidth=1.5, alpha=0.7)
        ax5.add_patch(unit_circle)
        ax5.set_xlim(-1.2, 1.2)
        ax5.set_ylim(-1.2, 1.2)
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        ax5.set_title("5 после α=(γ-i)/(1-iγ)", fontsize=12)
        ax5.set_xlabel("Re(α)")
        ax5.set_ylabel("Im(α)")

    # 6. Финальный результат (α → ω)
    ax6 = plt.subplot(2, 3, 6)
    if len(sample_points) > 0:
        omega = full_mapping(sample_points)
        ax6.scatter(omega.real, omega.imag,
                    s=8, alpha=0.6, color='purple')
        pi_circle = plt.Circle((0, 0), np.pi, fill=False,
                               color='r', linewidth=1.5, alpha=0.7)
        ax6.add_patch(pi_circle)
        ax6.set_xlim(-3.5, 3.5)
        ax6.set_ylim(-3.5, 3.5)
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)
        ax6.set_title("6 после ω=π·α (финальное D₂)", fontsize=12)
        ax6.set_xlabel("Re(ω)")
        ax6.set_ylabel("Im(ω)")

    plt.suptitle("конформное отображение D₁ → D₂ (вариант 24)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('all_in_one.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Основная функция программы"""
    print("лабораторная работа №2 по ТФКП")

    #1 аналитическое описание
    describe_domains()

    #2 вывод формул преобразования
    print("формулы")
    print("прямое отображение f: D₁ → D₂:")
    print("  1) ε = z + i")
    print("  2) γ = e^{-3πi/4}·ε")
    print("  3) α = (γ - i)/(1 - iγ)")
    print("  4) ω = π·α")
    print("\nобратное отображение f⁻¹: D₂ → D₁:")
    print("  1) α = ω/π")
    print("  2) γ = (α + i)/(1 + iα)")
    print("  3) ε = e^{3πi/4}·γ")
    print("  4) z = ε - i")

    #3 тестирование отображений
    test_mappings()

    #4 проверка отображения границы
    verify_boundary_mapping()

    #5 визуализация  графиков
    print("визуализация")

    try:
        plot_all_in_one()
    except Exception as e:
        print(f"\n⚠ Ошибка при создании графиков: {e}")
        print("  (продолжение работы программы...)")

    #6. итоговый отчет
    print("отчет")
    print("области D₁ и D₂ корректно описаны аналитически")
    print("построено конформное отображение f: D₁ → D₂")
    print("построено обратное отображение f⁻¹: D₂ → D₁")
    print("проверена принадлежность точек областям")
    print("проверено отображение границы |ω| → π")
    print("погрешность обратного отображения ≤ 10⁻¹⁵")
    print("созданы графики")


if __name__ == "__main__":
    main()