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

[create r5]
r5 is a indoor_room. "Room 5".

[create r6]
r6 is a indoor_room. "Room 6".

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "548540.0".
the Y of the a0r0 is "1542853.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548526.0".
the Y of the a1r0 is "1542914.4050772628".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "548626.0".
the Y of the a2r0 is "1542853.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "548604.5".
the Y of the a3r0 is "1542938.00390625".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "548271.0".
the Y of the a6r0 is "1543391.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "548302.0234020579".
the Y of the a7r0 is "1543417.0166743887".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "549112.0".
the Y of the a16r0 is "1543391.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "549092.3458787596".
the Y of the a17r0 is "1543415.7392271056".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "549096.4886295318".
the Y of the a40r0 is "1543485.2374336722".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "549104.4851666585".
the Y of the a41r0 is "1543459.4370599256".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "548261.0".
the Y of the a44r0 is "1543480.0".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "548292.350578465".
the Y of the a45r0 is "1543460.9916380297".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "548399.0".
the Y of the a54r0 is "1543139.0".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "548397.686594203".
the Y of the a55r0 is "1543089.6751110335".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "548970.2724759723".
the Y of the a75r0 is "1543448.39287269".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "548986.2746095151".
the Y of the a76r0 is "1543466.483492118".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "549066.6451298217".
the Y of the a77r0 is "1543443.8320928665".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "548315.9132133439".
the Y of the a81r0 is "1543440.2221422398".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "548558.9912057454".
the Y of the a82r0 is "1542964.2196130904".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "548592.5483636886".
the Y of the a83r0 is "1542966.911926578".

[create a93r0]
a93r0 is a area. "An area (93) in r0".
the printed name of the a93r0 is "Room 0".
Understand "Area 93 in Room 0" as a93r0.
the parent of the a93r0 is r0.
the X of the a93r0 is "548701.0415575999".
the Y of the a93r0 is "1543563.0990410426".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "548635.0".
the Y of the a20r1 is "1545514.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "548666.9940170939".
the Y of the a21r1 is "1545505.7505982907".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "549045.0".
the Y of the a22r1 is "1545514.0".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "549013.0059829059".
the Y of the a23r1 is "1545505.7505982907".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "549456.0".
the Y of the a34r1 is "1545424.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "549413.1798056157".
the Y of the a35r1 is "1545442.019978402".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "549446.0".
the Y of the a36r1 is "1545514.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "549415.1050372066".
the Y of the a37r1 is "1545487.7894485786".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "548501.0".
the Y of the a58r1 is "1545504.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "548534.2500000001".
the Y of the a59r1 is "1545448.4482064014".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "549000.239623524".
the Y of the a68r1 is "1545380.0019583767".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "549036.4173669468".
the Y of the a69r1 is "1545339.649859944".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "549398.1865386874".
the Y of the a72r1 is "1545459.6370670423".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "548679.760376476".
the Y of the a84r1 is "1545380.001958377".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "548591.9001965043".
the Y of the a88r1 is "1545193.9469272909".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "548620.4923510475".
the Y of the a89r1 is "1545322.8979623443".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547822.0".
the Y of the a24r2 is "1543403.0".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547778.8503889912".
the Y of the a25r2 is "1543461.502728187".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "547632.0".
the Y of the a56r2 is "1543369.0".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "547655.5661679944".
the Y of the a57r2 is "1543419.9341592467".

[create a62r2]
a62r2 is a area. "An area (62) in r2".
the printed name of the a62r2 is "Room 2".
Understand "Area 62 in Room 2" as a62r2.
the parent of the a62r2 is r2.
the X of the a62r2 is "547718.0091290362".
the Y of the a62r2 is "1543530.1696830266".

[create a63r2]
a63r2 is a area. "An area (63) in r2".
the printed name of the a63r2 is "Room 2".
Understand "Area 63 in Room 2" as a63r2.
the parent of the a63r2 is r2.
the X of the a63r2 is "547720.7606867261".
the Y of the a63r2 is "1543549.5866818968".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "546504.0".
the Y of the a18r3 is "1545514.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "546535.0876068376".
the Y of the a19r3 is "1545488.1662393163".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "546495.0".
the Y of the a32r3 is "1545424.0".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "546537.2698675498".
the Y of the a33r3 is "1545442.4980132454".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "547039.0".
the Y of the a60r3 is "1545504.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "547004.8999999999".
the Y of the a61r3 is "1545448.1052980134".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "546552.1884979304".
the Y of the a71r3 is "1545459.7787601026".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "546922.025775421".
the Y of the a90r3 is "1545326.639949738".

[create a91r3]
a91r3 is a area. "An area (91) in r3".
the printed name of the a91r3 is "Room 3".
Understand "Area 91 in Room 3" as a91r3.
the parent of the a91r3 is r3.
the X of the a91r3 is "546945.0735241219".
the Y of the a91r3 is "1545211.7267338363".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "547220.0".
the Y of the a4r4 is "1543391.0".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "547191.7559718895".
the Y of the a5r4 is "1543416.3336657637".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "546462.962966525".
the Y of the a10r4 is "1543486.1078364782".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "546453.4508685786".
the Y of the a11r4 is "1543459.4894136766".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "546511.2883186738".
the Y of the a12r4 is "1543395.7643013028".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "546475.5127742623".
the Y of the a13r4 is "1543416.765336723".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "547229.0".
the Y of the a14r4 is "1543480.0".

[create a15r4]
a15r4 is a area. "An area (15) in r4".
the printed name of the a15r4 is "Room 4".
Understand "Area 15 in Room 4" as a15r4.
the parent of the a15r4 is r4.
the X of the a15r4 is "547201.8491587281".
the Y of the a15r4 is "1543460.2680626004".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "547043.0".
the Y of the a50r4 is "1543391.0".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "547005.5884784686".
the Y of the a51r4 is "1543398.9327648387".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "546524.0".
the Y of the a52r4 is "1543372.0".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "546559.8594466473".
the Y of the a53r4 is "1543406.4633859482".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "546580.2923218387".
the Y of the a64r4 is "1543490.7715393982".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "546620.6355419154".
the Y of the a65r4 is "1543539.729419273".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "546997.4529379354".
the Y of the a74r4 is "1543454.7408583423".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "546489.360303978".
the Y of the a78r4 is "1543442.1314039843".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "547175.6662083615".
the Y of the a80r4 is "1543440.6920249432".

[create a92r4]
a92r4 is a area. "An area (92) in r4".
the printed name of the a92r4 is "Room 4".
Understand "Area 92 in Room 4" as a92r4.
the parent of the a92r4 is r4.
the X of the a92r4 is "546787.7500591346".
the Y of the a92r4 is "1543541.7980958915".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "546671.0".
the Y of the a26r5 is "1544638.0".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "546609.2809236948".
the Y of the a27r5 is "1544675.5".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "546671.0".
the Y of the a28r5 is "1545222.0".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "546592.1021505379".
the Y of the a29r5 is "1545187.1666666667".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "545111.0".
the Y of the a30r5 is "1545424.0".

[create a31r5]
a31r5 is a area. "An area (31) in r5".
the printed name of the a31r5 is "Room 5".
Understand "Area 31 in Room 5" as a31r5.
the parent of the a31r5 is r5.
the X of the a31r5 is "545153.0599999999".
the Y of the a31r5 is "1545442.5189999999".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "545120.0".
the Y of the a38r5 is "1545514.0".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "545150.9153780069".
the Y of the a39r5 is "1545488.1834621993".

[create a66r5]
a66r5 is a area. "An area (66) in r5".
the printed name of the a66r5 is "Room 5".
Understand "Area 66 in Room 5" as a66r5.
the parent of the a66r5 is r5.
the X of the a66r5 is "546530.6850543967".
the Y of the a66r5 is "1544725.221472193".

[create a67r5]
a67r5 is a area. "An area (67) in r5".
the printed name of the a67r5 is "Room 5".
Understand "Area 67 in Room 5" as a67r5.
the parent of the a67r5 is r5.
the X of the a67r5 is "546531.6040031037".
the Y of the a67r5 is "1545160.374629946".

[create a70r5]
a70r5 is a area. "An area (70) in r5".
the printed name of the a70r5 is "Room 5".
Understand "Area 70 in Room 5" as a70r5.
the parent of the a70r5 is r5.
the X of the a70r5 is "545168.0926719012".
the Y of the a70r5 is "1545459.8065726864".

[create a85r5]
a85r5 is a area. "An area (85) in r5".
the printed name of the a85r5 is "Room 5".
Understand "Area 85 in Room 5" as a85r5.
the parent of the a85r5 is r5.
the X of the a85r5 is "546127.1198796278".
the Y of the a85r5 is "1544395.8569714697".

[create a86r5]
a86r5 is a area. "An area (86) in r5".
the printed name of the a86r5 is "Room 5".
Understand "Area 86 in Room 5" as a86r5.
the parent of the a86r5 is r5.
the X of the a86r5 is "545526.3445626479".
the Y of the a86r5 is "1545334.3503250591".

[create a87r5]
a87r5 is a area. "An area (87) in r5".
the printed name of the a87r5 is "Room 5".
Understand "Area 87 in Room 5" as a87r5.
the parent of the a87r5 is r5.
the X of the a87r5 is "545531.2905415163".
the Y of the a87r5 is "1544443.0".

[create a8r6]
a8r6 is a area. "An area (8) in r6".
the printed name of the a8r6 is "Room 6".
Understand "Area 8 in Room 6" as a8r6.
the parent of the a8r6 is r6.
the X of the a8r6 is "545098.0".
the Y of the a8r6 is "1543391.0".

[create a9r6]
a9r6 is a area. "An area (9) in r6".
the printed name of the a9r6 is "Room 6".
Understand "Area 9 in Room 6" as a9r6.
the parent of the a9r6 is r6.
the X of the a9r6 is "545129.2137161086".
the Y of the a9r6 is "1543416.6339712918".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "545089.0".
the Y of the a42r6 is "1543480.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "545120.0059757694".
the Y of the a43r6 is "1543460.6579076622".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "545955.0".
the Y of the a46r6 is "1543391.0".

[create a47r6]
a47r6 is a area. "An area (47) in r6".
the printed name of the a47r6 is "Room 6".
Understand "Area 47 in Room 6" as a47r6.
the parent of the a47r6 is r6.
the X of the a47r6 is "545922.9473578264".
the Y of the a47r6 is "1543417.1323193451".

[create a48r6]
a48r6 is a area. "An area (48) in r6".
the printed name of the a48r6 is "Room 6".
Understand "Area 48 in Room 6" as a48r6.
the parent of the a48r6 is r6.
the X of the a48r6 is "545965.0".
the Y of the a48r6 is "1543480.0".

[create a49r6]
a49r6 is a area. "An area (49) in r6".
the printed name of the a49r6 is "Room 6".
Understand "Area 49 in Room 6" as a49r6.
the parent of the a49r6 is r6.
the X of the a49r6 is "545921.390839041".
the Y of the a49r6 is "1543462.3690068494".

[create a73r6]
a73r6 is a area. "An area (73) in r6".
the printed name of the a73r6 is "Room 6".
Understand "Area 73 in Room 6" as a73r6.
the parent of the a73r6 is r6.
the X of the a73r6 is "545906.8094615669".
the Y of the a73r6 is "1543444.8713538798".

[create a79r6]
a79r6 is a area. "An area (79) in r6".
the printed name of the a79r6 is "Room 6".
Understand "Area 79 in Room 6" as a79r6.
the parent of the a79r6 is r6.
the X of the a79r6 is "545143.389199255".
the Y of the a79r6 is "1543440.2597765361".

[create a94r6]
a94r6 is a area. "An area (94) in r6".
the printed name of the a94r6 is "Room 6".
Understand "Area 94 in Room 6" as a94r6.
the parent of the a94r6 is r6.
the X of the a94r6 is "545529.5".
the Y of the a94r6 is "1543572.0990566039".

[create d0]
d0 is a door. "A door between a93r0 and a88r1".
d0 is north of a93r0 and south of a88r1.

[create d1]
d1 is a door. "A door between a76r0 and a69r1".
d1 is north of a76r0 and south of a69r1.

[create d2]
d2 is a door. "A door between a88r1 and a63r2".
d2 is southwest of a88r1 and northeast of a63r2.

[create d4]
d4 is a door. "A door between a91r3 and a92r4".
d4 is south of a91r3 and north of a92r4.

[create d6]
d6 is a door. "A door between a63r2 and a91r3".
d6 is northwest of a63r2 and southeast of a91r3.

[create d10]
d10 is a door. "A door between a65r4 and a85r5".
d10 is northwest of a65r4 and southeast of a85r5.

[create d12]
d12 is a door. "A door between a87r5 and a94r6".
d12 is south of a87r5 and north of a94r6.

north of a0r0 is south of a1r0.

north of a2r0 is south of a3r0.

northwest of a4r4 is southeast of a5r4.

northeast of a6r0 is southwest of a7r0.

northeast of a8r6 is southwest of a9r6.

south of a10r4 is north of a11r4.

northwest of a12r4 is southeast of a13r4.

southwest of a14r4 is northeast of a15r4.

northwest of a16r0 is southeast of a17r0.

southeast of a18r3 is northwest of a19r3.

east of a20r1 is west of a21r1.

west of a22r1 is east of a23r1.

northwest of a24r2 is southeast of a25r2.

northwest of a26r5 is southeast of a27r5.

southwest of a28r5 is northeast of a29r5.

northeast of a30r5 is southwest of a31r5.

northeast of a32r3 is southwest of a33r3.

northwest of a34r1 is southeast of a35r1.

southwest of a36r1 is northeast of a37r1.

southeast of a38r5 is northwest of a39r5.

south of a40r0 is north of a41r0.

southeast of a42r6 is northwest of a43r6.

southeast of a44r0 is northwest of a45r0.

northwest of a46r6 is southeast of a47r6.

west of a48r6 is east of a49r6.

west of a50r4 is east of a51r4.

northeast of a52r4 is southwest of a53r4.

south of a54r0 is north of a55r0.

northeast of a56r2 is southwest of a57r2.

southeast of a58r1 is northwest of a59r1.

southwest of a60r3 is northeast of a61r3.

north of a62r2 is south of a63r2.

northeast of a64r4 is southwest of a65r4.

northwest of a27r5 is southeast of a66r5.

southwest of a29r5 is northeast of a67r5.

southeast of a68r1 is northwest of a69r1.

northeast of a31r5 is southwest of a70r5.

northeast of a33r3 is southwest of a71r3.

northwest of a35r1 is southeast of a72r1.

southwest of a37r1 is northeast of a72r1.

northwest of a47r6 is southeast of a73r6.

southwest of a49r6 is northeast of a73r6.

north of a51r4 is south of a74r4.

northeast of a75r0 is southwest of a76r0.

east of a76r0 is west of a77r0.

southwest of a64r4 is northeast of a78r4.

north of a53r4 is south of a64r4.

northwest of a25r2 is southeast of a62r2.

northeast of a9r6 is southwest of a79r6.

southeast of a11r4 is northwest of a78r4.

northeast of a13r4 is southwest of a78r4.

southwest of a15r4 is northeast of a80r4.

northwest of a17r0 is southeast of a77r0.

west of a41r0 is east of a77r0.

southeast of a43r6 is northwest of a79r6.

southeast of a45r0 is northwest of a81r0.

northeast of a1r0 is southwest of a82r0.

north of a3r0 is south of a83r0.

northwest of a5r4 is southeast of a80r4.

northeast of a7r0 is southwest of a81r0.

southeast of a39r5 is northwest of a70r5.

southeast of a19r3 is northwest of a71r3.

south of a21r1 is north of a84r1.

south of a23r1 is north of a68r1.

southwest of a66r5 is northeast of a85r5.

south of a86r5 is north of a87r5.

north of a66r5 is south of a67r5.

west of a67r5 is east of a86r5.

north of a88r1 is south of a89r1.

south of a90r3 is north of a91r3.

east of a70r5 is west of a86r5.

east of a69r1 is west of a72r1.

east of a65r4 is west of a92r4.

west of a85r5 is east of a87r5.

east of a74r4 is west of a80r4.

east of a79r6 is west of a94r6.

east of a81r0 is west of a93r0.

west of a73r6 is east of a94r6.

northwest of a74r4 is southeast of a92r4.

northwest of a75r0 is southeast of a93r0.

northeast of a57r2 is southwest of a62r2.

east of a82r0 is west of a83r0.

southwest of a75r0 is northeast of a83r0.

southeast of a55r0 is northwest of a82r0.

southwest of a84r1 is northeast of a89r1.

east of a71r3 is west of a90r3.

southeast of a59r1 is northwest of a89r1.

southwest of a61r3 is northeast of a90r3.

west of a68r1 is east of a84r1.

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
the player is in a10r4.

the orientation of the player is 6.
