import numpy as np
import math
import csv
from  utools import display_data
import matplotlib.pyplot as plt
from utools import smooth

windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
window_6hour = "window_6hour"  # 30 * 6 for a 2 minute sampling
window_hour = "window_hour"  # 30
window_halfhour = "window_1/2hour"  # 30
window_day = "window_day"  # 30 * 24
window_half_day = "window_half_day"  # 30 * 12
window_3days = "window_3days"  # 3 * 30 * 24


def smooth_span(dateTime, windowspan=window_day):
    span = windowspan

    # check if span is correct
    dt = dateTime[2] - dateTime[1]  # usually days
    if span == "window_6hour":  # 30 * 6 for a 2 minute sampling
        nspan = 6. / (dt * 24)
    elif span == "window_hour":  # 30 for a 2 minute sampling
        nspan = 1. / (dt * 24)
    elif span == "window_1/2hour":  # 30 for a 2 minute sampling
        nspan = 0.5 / (dt * 24)
    elif span == "window_day":  # 30 * 24 for a 2 minute sampling
        nspan = 24. / (dt * 24)
    elif span == "window_half_day":  # 30 * 12 for a 2 minute sampling
        nspan = 12. / (dt * 24)
    elif span == "window_3days":  # 3 * 30 * 24 for a 2 minute sampling
        nspan = 24. * 3 / (dt * 24)
    elif span == "window_7days":  # 7* 30 * 24 for a 2 minute sampling
        nspan = 24. * 7 / (dt * 24)

    return nspan

#######################################################################################################################
def wedderburnNumber(delta_rho, metaT, uSt, Ao, AvHyp_rho):
    """
    :param delta_rho:
    :param metaT:
    :param uSt:
    :param Ao:
    :param AvHyp_rho:
    :return:
    """
    """
    %----Author: Jordan S Read 2009 ----
    % updated 2 April 2010

    %Calculates the Wedderburn number for a particular system
    %using the following equation:
    %
    %   W = (g*delta_rho*(h^2))/(pHyp*(uSt^2)*Lo)
    %
    %where
    %   g = force of gravity
    %   delta_rho = density difference between the epilimnion and the
    %   hypolimnion
    %   metaT = thickness of the surface layer
    %   uSt = water friction velocity due to wind stress
    %   Lo = fetch length in the direction of the wind.
    %
    %Reference:
    % Imberger, Jorg, and John C. Patterson. "Physical Limnology." Advances
    % in Applied Mechanics 27 (1990): 314-317.
    """
    # Constants
    g = 9.81  # force of gravity

    Lo = 2 * math.sqrt(Ao / math.pi)  # Length at thermocline depth

    go = g * delta_rho / AvHyp_rho

    # Calculates W according to formula provided
    W = go * metaT ** 2 / (uSt ** 2 * Lo)


#######################################################################################################################
def buoyancyFreq(varL, wtr, thermoInd, depthAry):
    """
    --- Bogdan Hlevca 2016 ----
    :param varL:
    :param wtr:
    :param thermoInd:
    :param depthAry:
    :return:
    """
    # *** buoyancy frequency ***
    print('Calculating Buoyancy Frequency')
    g = 9.81  # m s - 1
    N2 = np.zero(varL)
    N2.fill(np.nan)

    for j in range(0, varL):
        Pw = waterDensity(wtr[j, thermoInd[j]], None)  # sal[j, thermoInd[j]])
        P2 = waterDensity(wtr[j, thermoInd(j) + 1], None)  # sal[j, thermoInd[j] + 1])
        dw = depthAry[thermoInd[j]]
        d2 = depthAry[thermoInd[j] + 1]
        N2[j] = g / Pw * (P2 - Pw) / (d2 - dw)
    return N2


#######################################################################################################################
def locPeaks(dataIn, dataMn):
    """
    :param dataIn:
    :param dataMn:
    :return:
    """
    """
    %----Author: Jordan S Read 2011 &  Bogdan Hlevca 2016 ----

    % this program attempts to mirror 'findpeaks.m' from the signal processing
    % toolbox

    % dataIn: vector of input data
    % dataMn: threshold for peak height

    % finds multiple peaks for dataIn
    % peaks: peak values
    % locs:  indices of peaks

    % -- description --
    % a peak is a peak if it represents a local maximum
    """

    if type(dataIn) is not np.array:
        print("dataIn is not of type numpy.array")
        return

    varL = len(dataIn)
    locs = np.zeros(varL)  # false(1,varL);
    peaks = np.empty(varL)
    peaks.fill(np.nan)

    for i in range(2, varL):
        [posPeak, pkI] = max(dataIn[i - 2:i + 1])
        if pkI == 2:
            peaks[i] = posPeak
            locs[i] = True

    inds = np.arange(1, varL)
    locs = inds[locs]
    peaks = peaks[locs]

    # remove all below threshold value
    useI = peaks >= dataMn
    peaks = peaks[useI]
    locs = locs[useI]

    return [peaks, locs]


#######################################################################################################################
def FindThermoDepth(rhoVar, depths, Smin=0.1, seasonal=False):
    """
    :param rhoVar:
    :param depths:
    :param Smin:
    :return:
    """
    """%----Author: Jordan S Read 2009  &  Bogdan Hlevca 2016 ----
    % updated 3 march 2011

    % removed signal processing toolbox function 'findpeaks.m' replaced with
    % 'locPeaks.m' which was written to provide the same functionality
    """

    dRhoPerc = 0.15  # min percentage max for unique thermocline step
    numDepths = len(depths)
    drho_dz = np.empty(numDepths-1)
    drho_dz.fill(np.nan)

    for i in range(0, numDepths-1):
        drho_dz[i] = (rhoVar[i + 1] - rhoVar[i]) / (depths[i + 1] - depths[i])

    if seasonal:
        # look for two distinct maximum slopes, lower one assumed to be seasonal
        mDrhoZ = np.max(drho_dz)  # find max slope
        thermoInd = np.argmax(drho_dz)
        thermoD = np.mean([depths[thermoInd], depths[thermoInd + 1]])  # depth of max slope
        if thermoInd > 0 and thermoInd < numDepths - 2:  # if within range,
            Sdn = -(depths[thermoInd + 1] - depths[thermoInd]) / (drho_dz[thermoInd + 1] - drho_dz[thermoInd])
            Sup = (depths[thermoInd] - depths[thermoInd - 1]) / (drho_dz[thermoInd] - drho_dz[thermoInd - 1])
            upD = depths(thermoInd)
            dnD = depths(thermoInd + 1)
            if not np.any([np.isinf(Sup), np.isinf(Sdn)]):
                thermoD = dnD * (Sdn / (Sdn + Sup)) + upD * (Sup / (Sdn + Sup))

        dRhoCut = np.max([dRhoPerc * mDrhoZ, Smin])
        [pks, locs] = locPeaks(drho_dz, dRhoCut)
        if pks.size == 0:  # isempty(pks)
            SthermoD = thermoD
            SthermoInd = thermoInd
        else:
            mDrhoZ = pks[len(pks)]
            SthermoInd = locs[len(pks)]
            if SthermoInd > thermoInd + 1:
                SthermoD = np.mean([depths[SthermoInd], depths(SthermoInd + 1)])
                if SthermoInd > 1 and SthermoInd < numDepths - 1:
                    Sdn = -(depths[SthermoInd + 1] - depths[SthermoInd]) / (
                        drho_dz[SthermoInd + 1] - drho_dz[SthermoInd])
                    Sup = (depths[SthermoInd] - depths[SthermoInd - 1]) / (
                        drho_dz[SthermoInd] - drho_dz[SthermoInd - 1])
                    upD = depths[SthermoInd]
                    dnD = depths[SthermoInd + 1]
                    if not np.any([np.isinf(Sup), np.isinf(Sdn)]):
                        SthermoD = dnD * (Sdn / (Sdn + Sup)) + upD * (Sup / (Sdn + Sup));
            else:
                SthermoD = thermoD
                SthermoInd = thermoInd
        if SthermoD < thermoD:
            SthermoD = thermoD
            SthermoInd = thermoInd
    else:
        mDrhoZ = np.max(np.abs(drho_dz))  # find max slope
        thermoInd = np.argmax(np.abs(drho_dz))
        thermoD = np.mean([depths[thermoInd], depths[thermoInd + 1]])  # depth of max slope
        if thermoInd > 0 and thermoInd < numDepths - 2:  # if within range,
            np.seterr(all='raise')
            Sdn = np.nan
            try:
                Sdn = -(depths[thermoInd + 1] - depths[thermoInd]) / (drho_dz[thermoInd + 1] - drho_dz[thermoInd])
            except:
                print("FindThermoDepth RuntimeError")

            Sup = (depths[thermoInd] - depths[thermoInd - 1]) / (drho_dz[thermoInd] - drho_dz[thermoInd - 1])
            upD = depths[thermoInd]
            dnD = depths[thermoInd + 1]
            if not np.isinf(Sup) and not np.isinf(Sdn):
                thermoD = dnD * (Sdn / (Sdn + Sup)) + upD * (Sup / (Sdn + Sup))

        SthermoD = thermoD
        SthermoInd = thermoInd

    return [thermoD, thermoInd, drho_dz, SthermoD, SthermoInd]


#######################################################################################################################
def FindMetaBot(drho_dz, thermoD, depths, slope):
    """
    :param drho_dz:   delta density /delta Z
    :param thermoD:  thermocline depth
    :param depths:   measured depths
    :param slope: d_rho/d_Z/m
    :return:
    """
    """
    %----Author: Jordan S Read 2009  &  Bogdan Hlevca 2016 ----
    %updated 12/15/2009 with thermoD pass
    """

    numDepths = len(depths)
    metaBot_depth = depths[numDepths - 1]  # default as bottom

    Tdepth = np.empty(numDepths-1)
    Tdepth.fill(np.nan)

    for i in range(0, numDepths - 1):
        Tdepth[i] = np.mean([depths[i + 1], depths[i]])

    ar = np.append(Tdepth, thermoD + 1e-6)
    sortDepth = np.sort(ar)
    sortInd = np.argsort(ar)
    drho_dz = np.interp(sortDepth, Tdepth, drho_dz)

    thermo_index = 0  # check this...
    thermoId = numDepths - 1

    for i in range(0, numDepths):
        if thermoId == sortInd[i]:
            thermo_index = i
            break

    ix = 0
    for i in range(thermo_index, numDepths):  # moving down from thermocline index
        if abs(drho_dz[i]) < slope:  # top of metalimnion
            metaBot_depth = sortDepth[i]
            break
        ix += 1

    if ix - thermo_index > 1 and drho_dz[thermo_index] > slope:
        metaBot_depth = np.interp(slope, drho_dz[thermo_index:ix], sortDepth[thermo_index:ix])

    if np.isnan(metaBot_depth):
        metaBot_depth = max(depths)

    return metaBot_depth


#######################################################################################################################
def FindMetaTop(drho_dz, thermoD, depths, slope):
    """
    :param drho_dz:   delta density /delta Z
    :param thermoD:  thermocline depth
    :param depths:   measured depths
    :param slope:    d_rho/d_Z/m
    :return:
    """
    """
    %----Author: Jordan S Read 2009  &  Bogdan Hlevca 2016 ----
    %updated 12/16/2009 with thermoD pass
    """
    numDepths = len(depths)
    metaTop_depth = np.mean([depths[1], depths[0]])

    Tdepth = np.empty(numDepths-1)
    Tdepth.fill(np.nan)

    for i in range(0, numDepths - 1):
        Tdepth[i] = np.mean([depths[i + 1], depths[i]])

    ar = np.append(Tdepth, thermoD + 1e-6)
    sortDepth = np.sort(ar)
    sortInd = np.argsort(ar)
    drho_dz = np.interp(sortDepth, Tdepth, drho_dz)

    thermo_index = 0
    thermoId = numDepths - 1
    for i in range(0, numDepths):
        if thermoId == sortInd[i]:
            thermo_index = i
            break

    ix=0
    for ix in range(thermo_index, 1, -1):  # moving up from thermocline index
        if abs(drho_dz[ix]) < slope:  # top of metalimnion
            metaTop_depth = sortDepth[ix]
            break

    # if interp can happen
    if thermo_index - ix > 1 and drho_dz[thermo_index] > slope:
        metaTop_depth = np.interp(slope, drho_dz[ix:thermo_index], sortDepth[ix:thermo_index])

    if np.isnan(metaTop_depth):
        metaTop_depth = min(depths)

    if metaTop_depth > thermoD:
        metaTop_depth = thermoD

    return metaTop_depth


#######################################################################################################################
def waterDensity(T, S=None):
    """
    :param T: temperature array
    :param S: salinity array
    :return:
    """
    """
    %% WDENSITY Density of water with supplied Temp and Salinity
    % rho = WDENSITY(Temperature, Salinity) is the density of water with the
    % given temperature and salinity
    % T is in �C
    % S is in Pratical Salinity Scale units (dimensionless)
    % T and S should be the same size, unless S is a scalar
    % rho is in grams/Liter
    %
    % <<<--- Effective range: 0-40�C, 0.5-43 Salinity --->>>
    %
    %----Author: Jordan S Read 2011 ----
    % Editied by:
    % 1/30/2010 - Luke Winslow <lawinslow@gmail.com>

    % three options: use all UNESCO, use all Martin & McCutcheon, or element
    % wise for both
    """

    def rhof(temp):
        T = temp.astype(np.float)
        return 1000 * (1 - (T + 288.9414) * (T - 3.9863) ** 2 / (508929.2 * (T + 68.12963)))

    MM = False
    UN = False
    elm = False
    if S is None:
        S = 0

    Trng = np.array([0, 40])
    Srng = np.array([0.5, 43])

    # check to see if all
    if np.all(np.greater(Srng[0], S)) and S != 0:  # (S < Srng[1]):
        MM = True  # use Martin & McCutcheon
    elif not (np.sum(np.less(T, Trng[0])) or np.sum(np.greater(T, Trng[1]))) and ( S == 0 \
            or not (np.sum(np.less(S, Srng[0])) or np.sum(np.greater(S, Srng[1])))):
        UN = True  # use UNESCO
    else:
        elm = True  # element-wise choice between M&M and UNESCO

    # use methods:
    if MM:
        """ <<equation provided by:
        % Martin, J.L., McCutcheon, S.C., 1999. Hydrodynamics and Transport
        % for Water Quality Modeling. Lewis Publications, Boca
        % Raton, FL, 794pp.  >>
        """

        if len(T) != 1:
            rho = [rhof(x) for x in T]
        else:
            rho = (1000 * (1 - (T + 288.9414) * (T - 3.9863) ** 2 / (508929.2 * (T + 68.12963))))

    elif UN:
        """
        % <<equations provided by:
        % Millero, F.J., Poisson, A., 1981. International one-atmosphere
        % equation of state of seawater. UNESCO Technical Papers in Marine
        % Science. No. 36     >>
        % --eqn (1):
        """
        rho_0 = 999.842594 + 6.793952e-2 * T - 9.095290e-3 * T ** 2 + 1.001685e-4 * T ** 3 - 1.120083e-6 * T ** 4 + \
                6.536335e-9 * T ** 5
        # --eqn (2):
        A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T ** 2 - 8.2467e-7 * T ** 3 + 5.3875e-9 * T ** 4
        # --eqn (3):
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2
        # --eqn (4):
        C = 4.8314e-4
        # --eqn (5):
        rho = rho_0 + A * S + B * S ** (3 / 2) + C * S
    else:
        rho = T * np.nan
        for j in range(1, len(T)):
            rho[j] = waterDensity(T[j], S[j])

    return rho


#######################################################################################################################
def layerDensity(top, bottom, wtr, depths, bthA, bthD, sal):
    """
    :param top: surface to top of metalimnion  [m]
    :param bottom: surface to bottom of metalimnion [m]
    :param wtr:    water temperature at depths [C]
    :param depths: depths where temperature was measured [m]
    :param bthA: bathymetry area
    :param bthD: bathymetry depth
    :param sal: salinity
    :return: aveDensity: average density of the epilimnion (kg/m^3)
    """
    """
    %----Author: Jordan S Read, 2009 & Bogdan Hlevca 2016 ----
    % updated 20 Nov 2011, adding timeseries bathymetric effects
    %
    %Finds the average density thermal layer, bounded by "top" and "bottom"
    % where distances are measured from the surface of the water column.
    %
    %Input:
    %   -temps: temperature values (celsius)
    %   -depths: corresponding depth values (m)
    %   -metaDef: critical slope used to define the metalimnion
    %   -top: surface to top of metalimnion
    %   -bottom: surface to bottom of metalimnion
    %
    %Output:
    %   -averageEpiDense: average density of the epilimnion (kg/m^3)
    %   -thermoDense: density at the thermocline (kg/m^3)
    """

    if sal == None:
        sal = wtr * 0

    if top > bottom:
        print('bottom depth must be greater than top')
        return None

    if len(wtr) != len(depths):
        print('water temperature array must be the same length as the depth array')
        return None

    # if bathymetry has negative values, intepolate to 0
    if min(bthD) < 0:
        useI = bthD >= 0
        if bthD != 0:
            depT = np.append(0, bthD[useI])
        else:
            depT = bthD[useI]

        bthA = np.interp(depT, bthD, bthA)
        bthD = depT

    dz = 0.1  # (m)

    numD = len(wtr)
    if max(bthD) > depths[numD - 1]:
        wtr[numD] = wtr[numD-1]
        # sal(numD+1) = sal(numD);
        depths[numD] = max(bthD)
    elif max(bthD) < depths[numD - 1]:
        bthD = np.append(bthD, depths[numD - 1])
        bthA = np.append(bthA, 0)

    if min(bthD) < depths[0]:
        wtr = np.append(wtr[0], wtr)
        # sal = [sal(0) sal]
        depths =  np.append(min(bthD), depths)

    Zo = np.min(depths)
    Io = np.argmin(depths)
    Ao = bthA[Io]
    if Ao == 0:
        print('Surface area cannot be zero, check *.bth file')
        return None

    # interpolates the bathymetry data
    if top == bottom:
        top = depths[len(depths)-2]
    layerD = np.arange(top, bottom, dz)
    layerT = np.interp(layerD, depths, wtr)
    layerS = None  # interp1(layerD,depths,sal)
    layerA = np.interp(layerD, bthD, bthA)
    layerP = waterDensity(layerT, layerS)

    mass = layerA * layerP * dz
    np.seterr(all='raise')
    aveDensity = np.nan
    try:
        aveDensity = np.sum(mass) / np.sum(layerA) / dz
    except:
        print("FindThermoDepth RuntimeError")


    return aveDensity


#######################################################################################################################
def schmidtStability(wtr, depths, bthA, bthD, sal=None):
    """
    :param wtr:            water temperature array
    :param depths:         depths of measured temperature
    :param bthA:           area of the lake at depth x
    :param bthD:           depth of the lake at depth x
    :param sal:            salinity
    :return:
    """
    """---Author: Jordan S Read 2009  &  Bogdan Hlevca 2016----
    % updated 7 March 2011, adding salinity
    % updated 20 Nov 2011, adding timeseries bathymetric effects

    %equation provided by:
    % Idso, S.B., 1973. On the concept of lake stability.
    % Limnol. Oceanogr. 18: 681-683.

    %   St = (g/A0)* [Integral from 0 to hm: (hv-h)*A(h)*p(h)*dh]

    % The program reverses the direction of this calculation so
    % that z = 0 is the surface, and positive z is downwards
    """

    if len(wtr) != len(depths):
        print(['water temperature array must be the same length as the depth array'])
    elif np.any(np.isnan(wtr)) or np.any(np.isnan(depths)) or np.any(np.isnan(bthA)) or np.any(np.isnan(bthD)):
        print('input arguments must be numbers')

    if type(bthD) is not np.ndarray:
        print("bthD is not of type numpy.array")
        return
    if type(bthA) is not np.ndarray:
        print("bthA is not of type numpy.array")
        return
    if type(wtr) is not np.ndarray:
        print("wtr is not of type numpy.array")
        return
    if type(depths) is not np.ndarray:
        print("sepths is not of type numpy.array")
        return

    if sal is not None:
        sal = wtr * 0

    g = 9.81  # (m s^-2)
    dz = 0.1  # (m)
    # if bathymetry has negative values, intepolate to 0
    if min(bthD) < 0:
        useI = bthD > 0  # bthD must be numpy.array
        if bthD != 0:
            depT = np.array([0, bthD[useI]])
        else:
            depT = bthD[useI]

        bthA = np.interp(depT, bthD, bthA)
        bthD = depT

    numD = len(wtr)
    if max(bthD) > depths[numD - 1]:
        wtr[numD + 1] = wtr[numD]
        sal[numD + 1] = sal[numD]
        depths[numD + 1] = max[bthD]
    elif max(bthD) < depths[numD - 1]:
        bthD = np.array([bthD, depths[numD]])
        bthA = np.array([bthA, 0])

    if min(bthD) < depths[1]:
        wtr = np.append(wtr[1], wtr)
        sal = None # np.append(sal[1], sal)
        depths = np.append(min(bthD), depths)

    Zo = np.min(depths)
    Io = np.argmin(depths)
    Zm = max(depths)
    Ao = bthA[Io]
    if Ao == 0:
        print('Surface area cannot be zero, check *.bth file')

    rhoL = waterDensity(wtr, sal)

    # interpolates the bathymetry data
    layerD = np.arange(Zo, Zm, dz)  # Zo:dz:Zm
    layerP = np.interp(layerD, depths, rhoL)
    layerA = np.interp(layerD, bthD, bthA)

    # find depth to the center of volume
    Zv = layerD * layerA * dz
    Zcv = np.sum(Zv) / np.sum(layerA) / dz

    numInt = len(layerA)

    st = np.empty(numInt)  # st = np.NaN(numInt, 1)
    st.fill(np.nan)

    for i in range(0, numInt):
        z = layerD[i]
        A = layerA[i]
        st[i] = -(Zcv - z) * layerP[i] * A * dz

    St = g / Ao * sum(st)
    return St


def uStar(wndSpeed, wndHeight, averageEpiDense):
    """
    :param wndSpeed:
    :param wndHeight:
    :param averageEpiDense: average density of epilimnion
    :return:
    """
    """
    %----Author: Jordan S Read 2009  &  Bogdan Hlevca 2016 ----
    % **** updated 19 Feb 2010 ****


    %equation (1) provided by:
    % Hicks, B.B., 1972. A procedure for the formulation of bulk transfer
    % coefficients over water bodies of different sizes. Boundary-Layer
    % Meterology 3: 201-213

    %equation (2) provided by:
    % Amorocho, J., DeVries, J.J., 1980. A new evaluation of the wind
    % stress coefficient over water surfaces. Journal of Geophysical
    % Research 85: 433-442.

    %equation (3) provided by:
    % Fischer, H.B., List, E.J., Koh, R.C.Y., Imberger, J., Brooks, N.H.,
    % 1979. Mixing in inland and coastal waters. Academic Press.

    %equation (4) provided by:
    % Imberger, J., 1985. The diurnal mixed layer. Limnology and Oceanography
    % 30: 737-770.
    """
    rhoAir = 1.2
    vonK = 0.4  # von Karman k

    # ~ eqn (1) ~
    if wndSpeed < 5:
        CD = 0.001
    else:
        CD = 0.0015

    # ~ eqn (2) ~
    # If the windspeed is not measured at a height of 10 meters, corrected by
    # (eqn 21) from Amorocho and DeVries, 1980
    if wndHeight != 10:
        wndSpeed = wndSpeed / (1 - np.sqrt(CD) / vonK * math.log(10 / wndHeight))

    # ~ eqn (3) ~
    tau = CD * rhoAir * wndSpeed ** 2

    # ~ eqn (4) ~
    if averageEpiDense is None:
        pass
    uStar = np.sqrt(tau / averageEpiDense)

    return uStar


#######################################################################################################################
def lakeNumber(bthA, bthD, uStar, St, metaT, metaB, rhoHyp):
    """
    :param bthA:             area of the lake at the depth x
    :param bthD:             depth (0 at top) at the depth x
    :param uStar:            friction velocity/shear velocity/ u star
    :param St:              Schmidt stability
    :param metaT:           depth to the top of metalimnion   - FindMetaTop
    :param metaB:           depth to the bottom of metalimnion -FindMetaBot
    :param rhoHyp:          density of the hypolimnion
    :return:
    """

    """%----Author: Jordan S Read 2009  &  Bogdan Hlevca 2016 ----

    % updated on 2 April 2010
    % updated on 25 January 2012

    %Calculates the lake number of a system using the
    %following equation:
    %
    %   Ln = (g*St*(1-(ht/hm)))/(p0*(uStar^2)*(A0^1.5)*(1-hv/hm)).

    %
    %References:
    %   -Imberger, Jorg, and John C. Patterson. "Physical Limnology." Advances in
    %    Applied Mechanics 27 (1990): 314-317.
    """

    if type(bthD) is not np.ndarray:
        print("bthD is not of type numpy.array")
        return
    if type(bthA) is not np.ndarray:
        print("bthA is not of type numpy.array")
        return

    g = 9.81
    dz = 0.1

    # if bathymetry has negative values, intepolate to 0
    if min(bthD) < 0:
        useI = bthD > 0
        if not bthD == 0:
            depT = np.array([0, bthD[useI]])
        else:
            depT = bthD[useI]

        bthA = np.interp(depT, bthD, bthA)
        bthD = depT

    Zo = np.min(bthD)
    Io = np.argmin(bthD)
    Ao = bthA[Io]
    if Ao == 0:
        print('Surface area cannot be zero, check *.bth file')

    # interpolates the bathymetry data
    layerD = np.arange(Zo, max(bthD), dz)
    layerA = np.interp(layerD, bthD, bthA)

    # find depth to the center of volume
    Zv = layerD * layerA * dz
    Zcv = np.sum(Zv) / np.sum(layerA) / dz  # should only need to do this once per
    # lake analyzer run...move out.
    St_uC = St * Ao / g
    # Calculates the Lake Number according to the formula provided
    if uStar == 0:
        uStar = 1e-3
    Ln = g * St_uC * (metaT + metaB) / (2 * rhoHyp * uStar ** 2 * Ao ** (3 / 2) * Zcv)
    print ("Ln = %f" % Ln)
    return Ln


#######################################################################################################################
def readConstants(filename):
    """
    :param filename:
    :return:
    """
    ifile = open(filename, 'rt')
    reader = csv.reader(ifile, delimiter=',', quotechar='"')

    outRs = maxZ = wndH = wndAv = lyrAv = outWn = wtrMx = wtrMn = wndMx = wndMn = drhDz = Tdiff = np.nan
    output = [outRs, maxZ, wndH, wndAv, lyrAv, outWn, wtrMx, wtrMn, wndMx, wndMn, drhDz, Tdiff]

    ix = 0
    for row in reader:
        try:
            if row[0][0] == "#":  # skip comments
                continue

            output[ix] = float(row[0])
            ix += 1
        except:
            print("readConstants error")
    return output


#######################################################################################################################
def readBathymetry(filename):
    """
    :param filename:
    :return:
    """
    ifile = open(filename, 'rt')
    reader = csv.reader(ifile, delimiter=',', quotechar='"')

    bthD = []
    bthA = []

    for row in reader:
        try:
            if row[0][0] == "#":  # skip comments
                continue

            bthD.append(float(row[0]))
            bthA.append(float(row[1]))
        except:
            print("readBathymetry error")

    bthD = np.array(bthD)
    bthA = np.array(bthA)

    output = [bthD, bthA]

    return output


#######################################################################################################################
def readWind(filename, resample=None):
    """
    :param filename:
    :return:
    """
    ifile = open(filename, 'rt')
    reader = csv.reader(ifile, delimiter=',', quotechar='"')

    date = []
    wndSpd = []
    wndD = None
    wnd = None

    for row in reader:
        try:
            if row[0][0] == "#":  # skip comments
                continue
            date.append(float(row[1]))
            wndSpd.append(float(row[2]))
        except:
            print("readWind error")

    if resample != None:
        pass
    wndD = np.array(date)
    wnd = np.array(wndSpd)

    output = [wndD, wnd]

    return output


#######################################################################################################################
def readWaterTemperature(filename, resample=None):
    """
    :param filename:
    :return:
    """
    ifile = open(filename, 'rt')
    reader = csv.reader(ifile, delimiter=',', quotechar='"')

    date = []
    wtrTemp = []
    head = []
    wtrD = None
    wtr = None
    wtrHead = None

    idx = 0
    for row in reader:
        try:
            if row[0][0] == "#":  # skip comments
                continue
            if idx == 0:
                head = row[1:]
            else:
                date.append(row[0])
                wtrTemp.append(row[1:])
        except:
            print("readWaterTemperature error")
        idx += 1

    if resample != None:
        pass
    wtrD = np.array(date)
    wtr = np.array(wtrTemp)
    wtrHead = np.array(head)
    output = [wtrD, wtr, wtrHead]

    return output


def calculateLakeNumber(path, constFileName, bthFileName, wndFileName, wtrFileName):
    # 1) *** read data ***
    print('Read data')
    Smin = 0.1

    [outRs, maxZ, wndH, wndAv, lyrAv, outWn, wtrMx, wtrMn, wndMx, wndMn, drhDz, Tdiff] = \
        readConstants(path + '/' + constFileName)
    """
    outRS = output resolution [s] ex 86400
    maxZ = manimum depth
    wndH = height at which wind was measured
    wndAv = time interval over which wind is averaged
    lyrAv = time interval over water layer temperatude is averaged
    outWnd = time interval for outlier
    wtrMx = max water temperature considered
    wtrMn = min water temperature considered
    wndMx = max wind speed considered
    wndMn = min wind speed considered
    drhDz = min metalimnion density slope (drho/dz per m)
    Tdiff = mixed temp differential
     """

    [bthD, bthA] = readBathymetry(path + '/' + bthFileName)
    """
    bthD = bathymetry depths
    bthA = bathymetry area
    """
    bthD = bthD.astype(np.float)
    bthA = bthA.astype(np.float)

    [wndD, wnd] = readWind(path + '/' + wndFileName)
    wndLength = len(wndD)
    """
    wndD = dates-times for wind
    wnd  = wind speed
    """
    wndD = wndD.astype(np.float)
    wnd = wnd.astype(np.float)

    [wtrD, wtr, wtrHead] = readWaterTemperature(path + '/' + wtrFileName)
    """
    wtrD = water dates
    wtr  = water temperature - matrix; each line has temperature date for all depths
    wtrHead = header for table which is the depth
    """
    wtrD = wtrD.astype(np.float)
    wtr = wtr.astype(np.float)

    [wtrLength, numDepths] = wtr.shape

    # for now lvl is constant
    lvlD = wtrD
    lvl = np.zeros(len(lvlD))
    lvl.fill(0)  # set to max level
    """
    lvlD - level date
    lvl  - levels >0 menas that is goinfg down to empty the lake
    """

    # 2) *** find layers ***
    print('Find Layers')
    Salinity = None
    varL = len(wtr)  # i.e. the number of time samples
    dates = wtrD[:]
    if drhDz < Smin:  # min
        Smin = drhDz

    depthAry = np.zeros(numDepths)  # NaN(1,numDepths);
    depthAry = wtrHead.astype(np.float)  # copy the depths for temperature from the table header
    mixed = np.zeros(varL)
    thermoD = np.ones(varL) * depthAry[numDepths - 1]
    metaT = thermoD
    metaB = metaT
    thermoInd = np.ones(varL)

    rho = np.zeros((varL, numDepths), dtype=float)
    for j in range(0, varL):
        rho[j, :] = waterDensity(wtr[j, :], S=Salinity)
        wtrT = wtr[j, :].astype(np.float)
        if wtrT.size >= 0 and np.abs(wtrT[0] - wtrT[len(wtrT) - 1]) > Tdiff:  # not mixed... # GIVES mixed if NaN!!!!
            # remove NaNs, need at least 3 values
            rhoT = rho[j, :]
            depT = depthAry
            depT = depT[~np.isnan(rhoT)] #depT[np.isnan(rhoT)] = []
            if len(depT) > 2:
                [thermoD[j], thermoInd[j], drho_dz, SthermoD, SthermoInd] = FindThermoDepth(rhoT, depT, Smin)
                metaT[j] = FindMetaTop(drho_dz, thermoD[j], depT, drhDz)
                metaB[j] = FindMetaBot(drho_dz, thermoD[j], depT, drhDz)
        else:
            mixed[j] = 1

    # 3) *** schmidt stability ***
    print('Calculating Schmidt Stability')
    St = np.zeros(varL)
    St.fill(np.nan)
    for j in range(0, varL):
        wtrT = wtr[j, :]
        salT = None  # sal[j,:]
        depT = depthAry
        depT = depT[~np.isnan(wtrT)]  # depT[np.isnan(wtrT)] = []
        # salT[(np.isnan(wtrT)] = []
        wtrT = wtrT[~np.isnan(wtrT)]  # wtrT[np.isnan(wtrT)] = []

        if len(wtrT) > 2:
            St[j] = schmidtStability(wtrT, depT, bthA, bthD, salT)

    nspan = smooth_span(dates, windowspan=window_day)
    sSt = smooth.smoothfit(dates, St, nspan, windows[1])['smoothed']
    display_data.display_one_temperature(dates, sSt, doy=True, ylabel='Schmidt stability [N m$^{-1}$]')
    plt.show()

    # 4) *** uStar ***
    print('Calculating U-star')
    uSt = np.zeros(varL)
    uSt.fill(np.nan)
    minDepth = depthAry[0]
    for j in range(0, varL):
        wtrT = wtr[j, :]
        salT = None  # sal[j,:]
        depT = depthAry
        depT = depT[~np.isnan(wtrT)]  # depT[np.isnan(wtrT)] = []
        # salT[np.isnan(wtrT)] = []
        wtrT = wtrT[~np.isnan(wtrT)]  # wtrT[np.isnan(wtrT)] = []
        if not np.any(np.isnan([minDepth, metaT[j]])) and wtrT.size > 0:
            AvEp_rho = layerDensity(minDepth, metaT[j], wtrT, depT, bthA, bthD, salT)
            uSt[j] = uStar(wnd[j], wndH, AvEp_rho)


    # 5) *** lake number ***
    print('Calculating Lake Number')
    Zm = max(bthD)
    LN = np.zeros(varL)
    LN.fill(np.nan)
    for j in range(0, varL):
        wtrT = wtr[j, :]
        salT = None  # sal(j,:)
        depT = depthAry
        depT = depT[~np.isnan(wtrT)] # depT[np.isnan(wtrT)] = []
        # salT (isnan(wtrT)) = [];
        wtrT = wtrT[~np.isnan(wtrT)]  # wtrT[np.isnan(wtrT)] = []
        if not np.any(np.isnan([Zm, metaB[j], St[j], uSt[j]])):
            AvHyp_rho = layerDensity(metaB[j], Zm, wtrT, depT, bthA, bthD, salT)
            LN[j] = lakeNumber(bthA, bthD, uSt[j], St[j], metaT[j], metaB[j], AvHyp_rho)

    nspan = smooth_span(dates, windowspan=window_day)
    sLN = smooth.smoothfit(dates, LN, nspan, windows[1])['smoothed']
    display_data.display_one_temperature(dates, sLN, doy=True, ylabel='Lake Number [non-dim.]')
    plt.show()

if __name__ == '__main__':
    path = '/home/bogdan/Documents/UofT/PhD/Data_Files/2013/LakeStability'
    constFileName = "Configuration.csv"
    bthFileName = "IH_Bathymetry.csv"
    wndFileName = "TH_Wind.csv"
    wtrFileName = "TH_WaterTemp.csv"
    calculateLakeNumber(path, constFileName, bthFileName, wndFileName, wtrFileName)
