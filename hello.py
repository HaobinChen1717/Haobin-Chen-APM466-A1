import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


TIME = [i / 2 for i in range(1, 11)]
TODAY = pd.to_datetime("2021/1/18")
df = pd.read_excel("A1_new.xlsx")
selected_bonds = df[(df["maturity date"] == "2021/3/1") | (df["maturity date"] == "2021/9/1") |
                    (df["maturity date"] == "2022/2/1") | (df["maturity date"] == "2022/8/1") |
                    (df["maturity date"] == "2023/3/1") | (df["maturity date"] == "2023/6/1") |
                    (df["maturity date"] == "2024/3/1") | (df["maturity date"] == "2024/9/1") |
                    (df["maturity date"] == "2025/3/1") |
                    (df["maturity date"] == "2025/9/1")].drop_duplicates(subset=["maturity date", "day"]). \
    reset_index(drop=True)

# Calculate Accrued interest
selected_bonds["maturity date"] = pd.to_datetime(selected_bonds["maturity date"])
selected_bonds['accrued interest'] = 0

# if Maturity is Mar or Sep or Feb or Aug, n = 128
selected_bonds['accrued interest'] = 128 / 365 * selected_bonds['coupon']

# Calculate Dirty price
selected_bonds['dirty price'] = selected_bonds['accrued interest'] + selected_bonds['price']

# Calculate YTM

class Coupon_Bond:
    def get_price(self, coupon, face_value, int_rate, years, freq=1):
        total_coupons_pv = self.get_coupons_pv(coupon, int_rate, years, freq)
        face_value_pv = self.get_face_value_pv(face_value, int_rate, years)
        result = total_coupons_pv + face_value_pv
        return result

    def get_face_value_pv(face_value, int_rate, years):
        fvpv = face_value / (1 + int_rate)**years
        return fvpv

    def get_coupons_pv(self, coupon, int_rate, years, freq=1):
        pv = 0
        for period in range(years*freq):
            pv += self.get_coupons_pv(coupon, int_rate, period+1, freq)
            return pv

    def get_coupon_pv(coupon, int_rate, period, freq):
        pv = coupon / (1 + int_rate/freq)**period
        return pv

    def get_ytm(self, bond_price, face_value, coupon, years, freq=1, estimate=0.05):
        import scipy
        from scipy import optimize
        get_yield = lambda int_rate: self.get_price(coupon, face_value, int_rate, years, freq) - bond_price
        return optimize.newton(get_yield, estimate)

coupon_bond_calculator = Coupon_Bond

# Plot YTM
plt.figure()
for i in range(1, 11):
    day = selected_bonds[selected_bonds["day"] == i]
    plt.plot(TIME, day["Yield"], marker='*')

plt.legend(["day " + str(i) for i in range(1, 11)])
plt.xlabel("time")
plt.ylabel("ytm")
plt.title('YTM Curve')

# Calculate spot rate
pre_sum = 0
pre_t = 0
r1 = 0
spot_rates = []
zcb_price = []

for index, row in selected_bonds.iterrows():
    if index % 10 == 0:
        pre_sum = 0
        pre_t = (row["maturity date"] - TODAY).days / 365
    if index != 0:
        TODAY += pd.DateOffset(1)
    t = (row["maturity date"] - TODAY).days / 365
    r = -math.log((row["dirty price"] - row["coupon"] * pre_sum / 2)/(100 + row["coupon"] / 2)) / t
    spot_rates.append(r)
    zcb_price.append(math.exp(-r * t))
    pre_sum += math.exp(-r * pre_t)
    pre_t = t

selected_bonds["spot rate"] = spot_rates
selected_bonds["zcb price"] = zcb_price

# Plot spot rate

plt.figure()
for i in range(1, 11):
    day = selected_bonds[selected_bonds["day"] == i]
    plt.plot(TIME, day["Spot rate"], marker='*')

plt.legend(["day " + str(i) for i in range(1, 11)])
plt.xlabel("time")
plt.ylabel("spot rate")
plt.title('spot rate Curve')
plt.show()

# Calculate forward rate

x = selected_bonds["NPER"]
y = selected_bonds["Spot rate"]
f11 = (y[3*10-8] * 2 - y[1*10-8] * 1) / (2 - 1)
f21 = (y[5*10-8] * 3 - y[1*10-8] * 1) / (3 - 1)
f31 = (y[7*10 - 8] * 4 - y[1 * 10 - 8] * 1) / (4 - 1)
f41 = (y[9*10-8] * 5 - y[1*10-8] * 1) / (5 - 1)

f1 = [f11, f21, f31, f41]

f12 = (y[3*10-7] * 2 - y[1*10-7] * 1) / (2 - 1)
f22 = (y[5*10-7] * 3 - y[1*10-7] * 1) / (3 - 1)
f32 = (y[7*10-7] * 4 - y[1*10-7] * 1) / (4 - 1)
f42 = (y[9*10-7] * 5 - y[1*10-7] * 1) / (5 - 1)

f2 = [f12, f22, f32, f42]
print(f2)


def forward_calculator(bond_info):
    y = spot_calculator(bond_info).squeeze()
    x, y = interpolation(bond_info["plot x"], y)
    f1 = (y[3] * 2 - y[1] * 1)/(2-1)
    f2 = (y[5] * 3 - y[1] * 1)/(3-1)
    f3 = (y[7] * 4 - y[1] * 1)/(4-1)
    f4 = (y[9] * 5 - y[1] * 1)/(5-1)
    f = [f1, f2, f3, f4]
    return f

# plot forward rate



axes = plt.gca()
axes.set_ylim([0.016, 0.02])
plt.legend(["day " + str(i) for i in range(1, 11)], loc="lower right", framealpha=0.3)
plt.xlabel("time")
plt.ylabel("rate")
plt.title('Forward Rate Curve')

# Calculate Cov(Xi) where Xi,j = log(ri,j+1/ri,j), j = 1,...,9
cov_mat1 = np.empty([9, 5])
for i in range(1, 10):
    all_ratio = np.log(selected_bonds.loc[selected_bonds['day'] == i + 1, ['Yield']].values.reshape(1, -1)[0] /
                       selected_bonds.loc[selected_bonds['day'] == i, ['Yield']].values.reshape(1, -1)[0])
    cov_mat1[i - 1] = all_ratio[1::2]

cov_mat2 = np.empty([1, 36])
ytm_cov = np.cov(cov_mat1.T)
eig_val1, eig_vec1 = np.linalg.eig(ytm_cov)
for i in range(36):
    cov_mat2[0][i] = forward_rates[i + 4] / forward_rates[i]

# eigenvalues and eigenvectors
ytm_cov = np.cov(cov_mat1.T)
eig_val1, eig_vec1 = np.linalg.eig(ytm_cov)

forward_cov = np.cov(cov_mat2.reshape(9, 4).T)
eig_val2, eig_vec2 = np.linalg.eig(forward_cov)

























