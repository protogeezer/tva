import numpy as np
import matplotlib.pyplot as plt

x = range(-4, 5)
xaxis = np.zeros(9)

def ci(aList):
    return [1.96*elt for elt in aList]

# elementary reading
female_on_female_beta_00 = [.00205404, .03846002, .09621856, 2.448e-16, 1.0083445, .40595879, .32331565, .25307384, .32077532]
female_on_female_se_00 = [.05966902, .0232847, .01143767, 0, .00814942, .00930746, .01024462, .01172091, .01691363]

male_on_female_beta_00 = [.04235613, -.01531709, .14078895, 1.006e-15, 1.0660366, .42312205, .32099026, .26707598, .36431408]
male_on_female_se_00 = [.06253506, .02472323, .01223259, 0, .00888978, .01015259, .01118517, .01278775, .01887035]

#elementary math (expect only small difference)
female_on_female_beta_01 = [.0413944, -.03120017, .0093687, -2.820e-16, 1.0780268, .42473587, .27228376, .21031136, .18418724]
female_on_female_se_01 = [.0404613, .01574322, .0075976, 0, .00508702, .00582521, .00643519, .00745525, .01081027]

male_on_female_beta_01 = [.09199758, -.05291108, .00803668, 3.328e-16, 1.0317622, .4099984, .26310724, .20856692, .18514955]
male_on_female_se_01 = [.03962109, .01529762, .00740926, 0, .00496452, .00567977, .00626864, .00725128, .01053644]

# middle reading (expect big effect)
# everything is crazy
female_on_female_beta_10 = [.02868972, .0910527, .08433868, 9.877e-16, .68208024, .70912946, .65510598, .41908727, -.8933489]
female_on_female_se_10 = [.02784017, .02168707, .02000949, 0, .01988286, .02716638, .04036149, .12066046, .50804053]

male_on_female_beta_10 = [.03998061, .01823437, .03870599, 1.446e-15, .77082486, .81847203, .87317131, .64896154, -.68367893]
male_on_female_se_10 = [.03373334, .02631144, .02420355, 0, .02402483, .03275792, .04813445, .13703116, .60781065]

# middle math
# don't find anything
female_on_female_beta_11 = [-.12379751, -.08622829, -.05297351, 1.348e-16, .97789483, .77738919, .60465997, .66984761, .12048069]
female_on_female_se_11 = [.0164246, .01268047, .01145594, 0, .01104553, .0153443, .02421431, .08204941, .45961139]

male_on_female_beta_11 = [-.08152404, -.0465499, -.00394235, -1.799e-16, .97933428, .7402372, .64746215, .51673339, .23987127]
male_on_female_se_11 = [.0165685, .01281584, .01155327, 0, .01107703, .01535734, .02387659, .07824616, .47388476]

plt.figure()
plt.title('Effect of VA on girls\' scores, middle school reading')
plt.errorbar(x, female_on_female_beta_11, yerr = ci(female_on_female_se_11), label = 'VA estimated from girls\' scores')
plt.errorbar(x, male_on_female_beta_11, yerr = ci(male_on_female_se_11), label = 'VA estimated from boys\' scores')
# x axis
plt.plot(x, np.zeros(len(x)), 'k')
plt.legend(loc=4)
plt.show()


#same = [-.13336531, -.05590078, .00988495, -3.685e-16, 1.0186062, .43678843, .29007515, .20392762, .2148752]
#other = [-.14026921, -.05579257, .01170434, 9.509e-17, .98109208, .41481424, .26967133, .2002731, .17555235]

#male_on_male_beta = [-0.1101688, -0.0431512, 0.0160163, 0, 0.9965238, 0.41284384, 0.26364097, 0.19194909, 0.17538434]
#male_on_male_se = [0.0135311,0.0086668,0.0055784,0, 0.0041598, 0.00489657, 0.00446279, 0.00652552, 0.0093883]

#female_on_male_beta = [-0.14086267, -0.05115036, 0.00487563, 0, 0.9835967, 0.39379423, 0.25410011, 0.1840348, 0.17715744]
#female_on_male_se = [0.01329267, 0.00862324, 0.00558034, 0, 0.00415486, 0.00489366, 0.00556929, 0.00654618, 0.00936961]

#difference_beta = [0.03069387, 0.00799916, 0.01114067, 0, 0.0129271, 0.01904961, 0.00954086, 0.00791429, -0.0017731]
#difference_se = [0.018968019, 0.0122259433, 0.0078904208, 0, 0.0058793535, 0.0069227383, 0.00713677, 0.0092430993, 0.0132638519]

#plt.figure()
#plt.errorbar(x, male_on_male_beta, yerr = male_on_male_se)
##plt.errorbar(x, male_on_male_beta, yerr = male_on_male_se)
##plt.errorbar(x, female_on_male_beta, yerr = female_on_male_se)
## plot x axis
#plt.errorbar(x, difference_beta, yerr = difference_se)
#plt.plot(x, same, label='Same gender')
#plt.plot(x, other, label= 'Other gender')
#plt.plot(x, xaxis)
#plt.title('Effect of teacher value-added on test scores over time')
#plt.legend()
#plt.show()



