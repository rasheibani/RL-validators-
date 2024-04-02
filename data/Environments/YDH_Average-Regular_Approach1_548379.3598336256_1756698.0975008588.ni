[define indoor_room]
indoor_room is a kind of room.
indoor_room has a text called description.
indoor_room has a text called printed name.

[define area]
area is a kind of room.
area has a text called description.
area has a text called printed name.
area has a indoor_room called parent.
area has a text called X.
area has a text called Y.
area can be enterable.
area is always enterable.

player has a number called orientation.


[define landmark]
landmark is a kind of thing.
landmark has a text called description.
landmark has a text called printed name.
landmark can be examined.
area has a list of landmark called visible_objects.

[create r0]
r0 is a indoor_room. "Room 0".

[create r1]
r1 is a indoor_room. "Room 1".

[create r2]
r2 is a indoor_room. "Room 2".

[create r3]
r3 is a indoor_room. "Room 3".

[create r4]
r4 is a indoor_room. "Room 4".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "546269.7072666292".
the Y of the a9r0 is "1758261.4777749956".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546563.8260530422".
the Y of the a10r0 is "1758217.5713468539".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546614.1177990806".
the Y of the a12r0 is "1758179.2133016966".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546930.9372773892".
the Y of the a13r0 is "1758139.7685073412".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "546932.4620894141".
the Y of the a19r0 is "1758141.1685851605".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "547990.0342558577".
the Y of the a20r0 is "1758201.1996321168".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "546522.0".
the Y of the a37r0 is "1758304.0".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "546554.8264604812".
the Y of the a38r0 is "1758305.7673539517".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "546223.0".
the Y of the a39r0 is "1758314.0".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "546253.8949627934".
the Y of the a40r0 is "1758287.7894485786".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "546982.1111111111".
the Y of the a53r0 is "1758304.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "546949.1923615086".
the Y of the a54r0 is "1758305.4040895503".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "546213.0".
the Y of the a69r0 is "1758224.0".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "546251.7029286583".
the Y of the a70r0 is "1758242.4774523713".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "548098.8043478262".
the Y of the a83r0 is "1758175.0".

[create a84r0]
a84r0 is a area. "An area (84) in r0".
the printed name of the a84r0 is "Room 0".
Understand "Area 84 in Room 0" as a84r0.
the parent of the a84r0 is r0.
the X of the a84r0 is "548062.0859683794".
the Y of the a84r0 is "1758182.79980989".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "548000.6455587936".
the Y of the a85r0 is "1758222.6529184869".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "548346.9647319478".
the Y of the a6r1 is "1758185.6648370589".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "548449.5422420469".
the Y of the a7r1 is "1758134.2164408192".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "548156.9071021571".
the Y of the a11r1 is "1758248.8198005701".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548094.0926719012".
the Y of the a14r1 is "1758259.8065726862".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "548463.7670482837".
the Y of the a15r1 is "1757293.1914207542".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "549854.46433213".
the Y of the a16r1 is "1757294.0".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "549867.5".
the Y of the a17r1 is "1758135.7211084904".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "548190.8791658573".
the Y of the a18r1 is "1758163.249220454".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "549510.1393831552".
the Y of the a21r1 is "1758259.8602906286".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "550224.8606168447".
the Y of the a22r1 is "1758259.8602906289".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "548808.8606168446".
the Y of the a23r1 is "1758259.8602906289".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "548046.0".
the Y of the a43r1 is "1758314.0".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "548076.9153780069".
the Y of the a44r1 is "1758288.1834621993".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "548857.0".
the Y of the a47r1 is "1758314.0".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "548825.9123931623".
the Y of the a48r1 is "1758288.166239316".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "549462.0".
the Y of the a49r1 is "1758314.0".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "549493.0876068377".
the Y of the a50r1 is "1758288.1662393163".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "550273.0".
the Y of the a51r1 is "1758314.0".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "550241.9123931624".
the Y of the a52r1 is "1758288.1662393163".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "548184.0".
the Y of the a55r1 is "1758194.0".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "548149.404245968".
the Y of the a56r1 is "1758212.0558052433".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "548037.0".
the Y of the a57r1 is "1758224.0".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "548079.06".
the Y of the a58r1 is "1758242.5189999999".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "548866.0".
the Y of the a59r1 is "1758224.0".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "548823.94".
the Y of the a60r1 is "1758242.5189999999".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "549453.0".
the Y of the a61r1 is "1758224.0".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "549495.06".
the Y of the a62r1 is "1758242.5189999999".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "550282.0".
the Y of the a63r1 is "1758224.0".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "550239.94".
the Y of the a64r1 is "1758242.5189999999".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "548098.8043478262".
the Y of the a83r1 is "1758175.0".

[create a0r2]
a0r2 is a area. "An area (0) in r2".
the printed name of the a0r2 is "Room 2".
Understand "Area 0 in Room 2" as a0r2.
the parent of the a0r2 is r2.
the X of the a0r2 is "549867.5".
the Y of the a0r2 is "1756368.1939858492".

[create a1r2]
a1r2 is a area. "An area (1) in r2".
the printed name of the a1r2 is "Room 2".
Understand "Area 1 in Room 2" as a1r2.
the parent of the a1r2 is r2.
the X of the a1r2 is "550233.8279210806".
the Y of the a1r2 is "1756245.02905727".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "548379.3598336256".
the Y of the a24r2 is "1756698.0975008588".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "548450.9999999998".
the Y of the a25r2 is "1756368.5778301884".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "548191.3996810535".
the Y of the a27r2 is "1756558.6289927168".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "548428.3743961761".
the Y of the a28r2 is "1756894.9239713205".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "548817.7817693064".
the Y of the a29r2 is "1756244.9748289345".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "546546.389199255".
the Y of the a31r2 is "1756240.2597765364".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "546930.1865194304".
the Y of the a32r2 is "1756370.3208574704".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "549485.389199255".
the Y of the a33r2 is "1756240.2597765364".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "548068.9380132287".
the Y of the a34r2 is "1756248.6127976696".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "547955.7930517282".
the Y of the a35r2 is "1756336.7527502163".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "547971.1439781914".
the Y of the a36r2 is "1756343.4433771".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "550282.0".
the Y of the a71r2 is "1756191.0".

[create a72r2]
a72r2 is a area. "An area (72) in r2".
the printed name of the a72r2 is "Room 2".
Understand "Area 72 in Room 2" as a72r2.
the parent of the a72r2 is r2.
the X of the a72r2 is "550250.9579422104".
the Y of the a72r2 is "1756216.6166125853".

[create a75r2]
a75r2 is a area. "An area (75) in r2".
the printed name of the a75r2 is "Room 2".
Understand "Area 75 in Room 2" as a75r2.
the parent of the a75r2 is r2.
the X of the a75r2 is "548875.0".
the Y of the a75r2 is "1756280.0".

[create a76r2]
a76r2 is a area. "An area (76) in r2".
the printed name of the a76r2 is "Room 2".
Understand "Area 76 in Room 2" as a76r2.
the parent of the a76r2 is r2.
the X of the a76r2 is "548832.1485252036".
the Y of the a76r2 is "1756261.855767114".

[create a77r2]
a77r2 is a area. "An area (77) in r2".
the printed name of the a77r2 is "Room 2".
Understand "Area 77 in Room 2" as a77r2.
the parent of the a77r2 is r2.
the X of the a77r2 is "550291.0".
the Y of the a77r2 is "1756280.0".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "550248.1485252037".
the Y of the a78r2 is "1756261.8557671143".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "548615.0".
the Y of the a81r2 is "1756827.0".

[create a82r2]
a82r2 is a area. "An area (82) in r2".
the printed name of the a82r2 is "Room 2".
Understand "Area 82 in Room 2" as a82r2.
the parent of the a82r2 is r2.
the X of the a82r2 is "548549.7360487827".
the Y of the a82r2 is "1756865.2744186048".

[create a86r2]
a86r2 is a area. "An area (86) in r2".
the printed name of the a86r2 is "Room 2".
Understand "Area 86 in Room 2" as a86r2.
the parent of the a86r2 is r2.
the X of the a86r2 is "546501.0".
the Y of the a86r2 is "1756191.0".

[create a87r2]
a87r2 is a area. "An area (87) in r2".
the printed name of the a87r2 is "Room 2".
Understand "Area 87 in Room 2" as a87r2.
the parent of the a87r2 is r2.
the X of the a87r2 is "546532.2137161084".
the Y of the a87r2 is "1756216.633971292".

[create a88r2]
a88r2 is a area. "An area (88) in r2".
the printed name of the a88r2 is "Room 2".
Understand "Area 88 in Room 2" as a88r2.
the parent of the a88r2 is r2.
the X of the a88r2 is "549440.0".
the Y of the a88r2 is "1756191.0".

[create a89r2]
a89r2 is a area. "An area (89) in r2".
the printed name of the a89r2 is "Room 2".
Understand "Area 89 in Room 2" as a89r2.
the parent of the a89r2 is r2.
the X of the a89r2 is "549471.2137161084".
the Y of the a89r2 is "1756216.633971292".

[create a94r2]
a94r2 is a area. "An area (94) in r2".
the printed name of the a94r2 is "Room 2".
Understand "Area 94 in Room 2" as a94r2.
the parent of the a94r2 is r2.
the X of the a94r2 is "546492.0".
the Y of the a94r2 is "1756280.0".

[create a95r2]
a95r2 is a area. "An area (95) in r2".
the printed name of the a95r2 is "Room 2".
Understand "Area 95 in Room 2" as a95r2.
the parent of the a95r2 is r2.
the X of the a95r2 is "546523.0059757693".
the Y of the a95r2 is "1756260.6579076622".

[create a96r2]
a96r2 is a area. "An area (96) in r2".
the printed name of the a96r2 is "Room 2".
Understand "Area 96 in Room 2" as a96r2.
the parent of the a96r2 is r2.
the X of the a96r2 is "549431.0".
the Y of the a96r2 is "1756280.0".

[create a97r2]
a97r2 is a area. "An area (97) in r2".
the printed name of the a97r2 is "Room 2".
Understand "Area 97 in Room 2" as a97r2.
the parent of the a97r2 is r2.
the X of the a97r2 is "549462.0059757694".
the Y of the a97r2 is "1756260.6579076622".

[create a98r2]
a98r2 is a area. "An area (98) in r2".
the printed name of the a98r2 is "Room 2".
Understand "Area 98 in Room 2" as a98r2.
the parent of the a98r2 is r2.
the X of the a98r2 is "548866.0".
the Y of the a98r2 is "1756191.0".

[create a99r2]
a99r2 is a area. "An area (99) in r2".
the printed name of the a99r2 is "Room 2".
Understand "Area 99 in Room 2" as a99r2.
the parent of the a99r2 is r2.
the X of the a99r2 is "548834.7862838916".
the Y of the a99r2 is "1756216.6339712918".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "545983.0".
the Y of the a2r3 is "1756368.5778301887".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "546349.7290448344".
the Y of the a3r3 is "1756245.062703054".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "545978.1730804583".
the Y of the a26r3 is "1757059.7005024818".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "545601.3945333334".
the Y of the a30r3 is "1756240.2686666667".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "546398.0".
the Y of the a73r3 is "1756191.0".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "546366.7862838916".
the Y of the a74r3 is "1756216.633971292".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "546407.0".
the Y of the a79r3 is "1756280.0".

[create a80r3]
a80r3 is a area. "An area (80) in r3".
the printed name of the a80r3 is "Room 3".
Understand "Area 80 in Room 3" as a80r3.
the parent of the a80r3 is r3.
the X of the a80r3 is "546363.9383292025".
the Y of the a80r3 is "1756261.8770228897".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "545556.0".
the Y of the a90r3 is "1756191.0".

[create a91r3]
a91r3 is a area. "An area (91) in r3".
the printed name of the a91r3 is "Room 3".
Understand "Area 91 in Room 3" as a91r3.
the parent of the a91r3 is r3.
the X of the a91r3 is "545587.2137161086".
the Y of the a91r3 is "1756216.633971292".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "545547.0".
the Y of the a92r3 is "1756280.0".

[create a93r3]
a93r3 is a area. "An area (93) in r3".
the printed name of the a93r3 is "Room 3".
Understand "Area 93 in Room 3" as a93r3.
the parent of the a93r3 is r3.
the X of the a93r3 is "545577.8730024239".
the Y of the a93r3 is "1756260.6444609193".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "545409.1530110595".
the Y of the a4r4 is "1758137.6714275307".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "545791.915169287".
the Y of the a5r4 is "1758261.610847683".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "545044.0175949464".
the Y of the a8r4 is "1758261.5825998154".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "544997.0".
the Y of the a41r4 is "1758314.0".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "545027.9153780065".
the Y of the a42r4 is "1758288.1834621995".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "545839.0".
the Y of the a45r4 is "1758314.0".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "545807.9123931623".
the Y of the a46r4 is "1758288.1662393163".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "544988.0".
the Y of the a65r4 is "1758224.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "545026.2466592428".
the Y of the a66r4 is "1758242.9003340756".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "545848.0".
the Y of the a67r4 is "1758224.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "545808.9448424068".
the Y of the a68r4 is "1758242.8194842408".

[create d0]
d0 is a door. "A door between a13r0 and a32r2".
d0 is south of a13r0 and north of a32r2.

[create d2]
d2 is a door. "A door between a16r1 and a0r2".
d2 is south of a16r1 and north of a0r2.

[create d3]
d3 is a door. "A door between a15r1 and a28r2".
d3 is south of a15r1 and north of a28r2.

[create d6]
d6 is a door. "A door between a12r0 and a26r3".
d6 is southwest of a12r0 and northeast of a26r3.

[create d8]
d8 is a door. "A door between a26r3 and a4r4".
d8 is northwest of a26r3 and southeast of a4r4.

east of a0r2 is west of a1r2.

east of a2r3 is west of a3r3.

east of a4r4 is west of a5r4.

southeast of a6r1 is northwest of a7r1.

west of a4r4 is east of a8r4.

east of a9r0 is west of a10r0.

west of a6r1 is east of a11r1.

east of a12r0 is west of a13r0.

southeast of a10r0 is northwest of a12r0.

west of a11r1 is east of a14r1.

east of a15r1 is west of a16r1.

north of a16r1 is south of a17r1.

south of a7r1 is north of a15r1.

west of a6r1 is east of a18r1.

east of a19r0 is west of a20r0.

west of a17r1 is east of a21r1.

east of a17r1 is west of a22r1.

east of a7r1 is west of a23r1.

south of a24r2 is north of a25r2.

north of a2r3 is south of a26r3.

southwest of a24r2 is northeast of a27r2.

north of a24r2 is south of a28r2.

east of a25r2 is west of a29r2.

west of a2r3 is east of a30r3.

east of a31r2 is west of a32r2.

west of a0r2 is east of a33r2.

west of a25r2 is east of a34r2.

east of a32r2 is west of a35r2.

southwest of a27r2 is northeast of a36r2.

east of a37r0 is west of a38r0.

southeast of a39r0 is northwest of a40r0.

southeast of a41r4 is northwest of a42r4.

southeast of a43r1 is northwest of a44r1.

southwest of a45r4 is northeast of a46r4.

southwest of a47r1 is northeast of a48r1.

southeast of a49r1 is northwest of a50r1.

southwest of a51r1 is northeast of a52r1.

west of a53r0 is east of a54r0.

northwest of a55r1 is southeast of a56r1.

northeast of a57r1 is southwest of a58r1.

northwest of a59r1 is southeast of a60r1.

northeast of a61r1 is southwest of a62r1.

northwest of a63r1 is southeast of a64r1.

northeast of a65r4 is southwest of a66r4.

northwest of a67r4 is southeast of a68r4.

northeast of a69r0 is southwest of a70r0.

northwest of a71r2 is southeast of a72r2.

northwest of a73r3 is southeast of a74r3.

southwest of a75r2 is northeast of a76r2.

southwest of a77r2 is northeast of a78r2.

southwest of a79r3 is northeast of a80r3.

northwest of a81r2 is southeast of a82r2.

west of a83r0 is east of a84r0.

northeast of a86r2 is southwest of a87r2.

northeast of a88r2 is southwest of a89r2.

northeast of a90r3 is southwest of a91r3.

southeast of a92r3 is northwest of a93r3.

southeast of a94r2 is northwest of a95r2.

southeast of a96r2 is northwest of a97r2.

northwest of a98r2 is southeast of a99r2.

north of a18r1 is south of a55r1.

southwest of a31r2 is northeast of a87r2.

southwest of a33r2 is northeast of a89r2.

southwest of a30r3 is northeast of a91r3.

northwest of a30r3 is southeast of a93r3.

northwest of a31r2 is southeast of a95r2.

northwest of a33r2 is southeast of a97r2.

southeast of a29r2 is northwest of a99r2.

southeast of a1r2 is northwest of a72r2.

southeast of a3r3 is northwest of a74r3.

northeast of a22r1 is southwest of a52r1.

north of a19r0 is south of a54r0.

north of a10r0 is south of a38r0.

southeast of a5r4 is northwest of a68r4.

southwest of a9r0 is northeast of a70r0.

northwest of a9r0 is southeast of a40r0.

northwest of a8r4 is southeast of a42r4.

northwest of a14r1 is southeast of a44r1.

northeast of a5r4 is southwest of a46r4.

northeast of a23r1 is southwest of a48r1.

northwest of a21r1 is southeast of a50r1.

northeast of a20r0 is southwest of a85r0.

south of a11r1 is north of a56r1.

northeast of a13r0 is southwest of a19r0.

southwest of a14r1 is northeast of a58r1.

southeast of a23r1 is northwest of a60r1.

southwest of a21r1 is northeast of a62r1.

southeast of a22r1 is northwest of a64r1.

southwest of a8r4 is northeast of a66r4.

northeast of a29r2 is southwest of a76r2.

northeast of a1r2 is southwest of a78r2.

northeast of a3r3 is southwest of a80r3.

northeast of a35r2 is southwest of a36r2.

northwest of a34r2 is southeast of a36r2.

east of a28r2 is west of a82r2.

east of a20r0 is west of a84r0.

Definition: a direction (called thatway) is viable if the room thatway from the location is not nowhere.

nLooking is a number that varies. nLooking is 0. 
dirNumber is a number that varies.
dirNumber is 0. 
relDirDesc is a text that varies. 

[describe areas and rooms]
After looking:
	now nLooking is 1; 
	let accessibleRooms be a list of rooms;
	let accessibleAreas be a list of areas;
	let pDirections be list of viable directions;
	let parentSource be the parent of the location of player;
	let relDirs be a list of number;
	repeat with dirToLookAt running through pDirections:
		try silently going dirToLookAt;
		if rule succeeded:
			now dirNumber is 0;
			if "[dirToLookAt]" is "south":
				now dirNumber is 4;
			if "[dirToLookAt]" is "east":
				now dirNumber is 6;
			if "[dirToLookAt]" is "west":
				now dirNumber is 2;
			if "[dirToLookAt]" is "northwest":
				now dirNumber is 1;
			if "[dirToLookAt]" is "southwest":
				now dirNumber is 3;
			if "[dirToLookAt]" is "southeast":
				now dirNumber is 5;
			if "[dirToLookAt]" is "northeast":
				now dirNumber is 7;
			let relDir be the remainder after dividing the orientation of the player - dirNumber + 80 by 8;
			add relDir to relDirs;
			if relDir is 0:
				now relDirDesc is "at the front";
			if relDir is 1:
				now relDirDesc is "at the slight right";
			if relDir is 2:
				now relDirDesc is "at the right";
			if relDir is 3:
				now relDirDesc is "at the sharp right";
			if relDir is 4:
				now relDirDesc is "at the back";
			if relDir is 5:
				now relDirDesc is "at the sharp left";
			if relDir is 6:
				now relDirDesc is "at the left";
			if relDir is 7:
				now relDirDesc is "at the slight left";
			let destinationParent be the parent of the location of the player;
			if "[parentSource]" is "[destinationParent]":
				say "You can continue in the [parentSource] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the location of the player to accessibleAreas;
			otherwise:
				say "You can enter the [destinationParent] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the destinationParent to accessibleRooms;
			try silently going the opposite of dirToLookAt;
	repeat with vO running through the visible_objects of the location:
		say "[vO] is visible from here, but too far! You can move in this room to examine or access [it]";
	now nLooking is 0.
Before going through a locked door when nLooking is 1:
	stop the action.
Going front is an action applying to nothing. Understand "go front" as going front.

Check going front:
	if the orientation of the player is:
		-- 0: try the player going north;
		-- 1: try the player going northwest;
		-- 2: try the player going west;
		-- 3: try the player going southwest;
		-- 4: try the player going south;
		-- 5: try the player going southeast;
		-- 6: try the player going east;
		-- 7: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going back is an action applying to nothing. Understand "go back" as going back.

Check going back:
	if the orientation of the player is:
		-- 4: try the player going north;
		-- 5: try the player going northwest;
		-- 6: try the player going west;
		-- 7: try the player going southwest;
		-- 0: try the player going south;
		-- 1: try the player going southeast;
		-- 2: try the player going east;
		-- 3: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".



Going left is an action applying to nothing. Understand "go left" as going left.

Check going left:
	if the orientation of the player is:
		-- 6: try the player going north;
		-- 7: try the player going northwest;
		-- 0: try the player going west;
		-- 1: try the player going southwest;
		-- 2: try the player going south;
		-- 3: try the player going southeast;
		-- 4: try the player going east;
		-- 5: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going right is an action applying to nothing. Understand "go right" as going right.

Check going right:
	if the orientation of the player is:
		-- 2: try the player going north;
		-- 3: try the player going northwest;
		-- 4: try the player going west;
		-- 5: try the player going southwest;
		-- 6: try the player going south;
		-- 7: try the player going southeast;
		-- 0: try the player going east;
		-- 1: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Going sharp right is an action applying to nothing. Understand "go sharp right" as going sharp right.
Going slight right is an action applying to nothing. Understand "go slight right" as going slight right.
Going sharp left is an action applying to nothing. Understand "go sharp left" as going sharp left.
Going slight left is an action applying to nothing. Understand "go slight left" as going slight left.

Check going sharp right:
	if the orientation of the player is:
		-- 3: try the player going north;
		-- 4: try the player going northwest;
		-- 5: try the player going west;
		-- 6: try the player going southwest;
		-- 7: try the player going south;
		-- 0: try the player going southeast;
		-- 1: try the player going east;
		-- 2: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight right:
	if the orientation of the player is:
		-- 1: try the player going north;
		-- 2: try the player going northwest;
		-- 3: try the player going west;
		-- 4: try the player going southwest;
		-- 5: try the player going south;
		-- 6: try the player going southeast;
		-- 7: try the player going east;
		-- 0: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going sharp left:
	if the orientation of the player is:
		-- 5: try the player going north;
		-- 6: try the player going northwest;
		-- 7: try the player going west;
		-- 0: try the player going southwest;
		-- 1: try the player going south;
		-- 2: try the player going southeast;
		-- 3: try the player going east;
		-- 4: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight left:
	if the orientation of the player is:
		-- 7: try the player going north;
		-- 0: try the player going northwest;
		-- 1: try the player going west;
		-- 2: try the player going southwest;
		-- 3: try the player going south;
		-- 4: try the player going southeast;
		-- 5: try the player going east;
		-- 6: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Instead of going north:
	if nLooking is 0:
		now the orientation of the player is 0;
	continue the action.

Instead of going south:
	if nLooking is 0:
		now the orientation of the player is 4;
	continue the action.

Instead of going east:
	if nLooking is 0:
		now the orientation of the player is 6;
	continue the action.

Instead of going west:
	if nLooking is 0:
		now the orientation of the player is 2;
	continue the action.

Instead of going northwest:
	if nLooking is 0:
		now the orientation of the player is 1;
	continue the action.

Instead of going southwest:
	if nLooking is 0:
		now the orientation of the player is 3;
	continue the action.

Instead of going northeast:
	if nLooking is 0:
		now the orientation of the player is 7;
	continue the action.

Instead of going southeast:
	if nLooking is 0:
		now the orientation of the player is 5;
	continue the action.

Understand "veer left" as going slight left.
Understand "veer right" as going slight right.
Understand "turn left" as going left.
Understand "turn right" as going right.
Understand "turn around" as going back.
Understand "turn sharp left" as going sharp left.
Understand "Go straight" as going front.
Understand "turn sharp right" as going sharp right.
Understand "Turn slightly left" as going slight left.
Understand "Turn slightly right" as going slight right.
Understand "Make a sharp right" as going sharp right.
Understand "Make a sharp left" as going sharp left.
Every turn:
	let xx be the X of the location of the player;
	let yy be the Y of the location of the player;
	say "X: [xx][line break]";
	say "Y: [yy]".
the player is in a24r2.

the orientation of the player is 6.
