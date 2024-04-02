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
the X of the a0r0 is "546111.9912057454".
the Y of the a0r0 is "1478364.2196130904".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "546145.5483636888".
the Y of the a1r0 is "1478366.9119265778".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545950.686594203".
the Y of the a3r0 is "1478489.6751110335".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "546093.0".
the Y of the a41r0 is "1478253.0".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "546079.0".
the Y of the a42r0 is "1478314.4050772628".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "546179.0".
the Y of the a43r0 is "1478253.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "546157.5".
the Y of the a44r0 is "1478338.00390625".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "545952.0".
the Y of the a51r0 is "1478539.0".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546561.3737630215".
the Y of the a2r1 is "1479989.4484652618".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545120.0".
the Y of the a13r1 is "1480914.0".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545150.9153780069".
the Y of the a14r1 is "1480888.1834621993".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "547000.1111111111".
the Y of the a19r1 is "1480904.0".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "546967.1923615087".
the Y of the a20r1 is "1480905.4040895503".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "546188.0".
the Y of the a21r1 is "1480914.0".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "546219.9940170939".
the Y of the a22r1 is "1480905.7505982907".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546513.0".
the Y of the a23r1 is "1480914.0".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546544.9940170939".
the Y of the a24r1 is "1480905.7505982907".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "546250.0".
the Y of the a27r1 is "1479660.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546213.2".
the Y of the a28r1 is "1479750.5878684372".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "545111.0".
the Y of the a29r1 is "1480824.0".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "545153.0599999999".
the Y of the a30r1 is "1480842.5189999999".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547346.0".
the Y of the a35r1 is "1480824.0".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "547305.6686930092".
the Y of the a36r1 is "1480842.2965214455".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "547336.0".
the Y of the a39r1 is "1480914.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547305.1050372064".
the Y of the a40r1 is "1480887.7894485784".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "546563.3747130401".
the Y of the a59r1 is "1480724.7007432287".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547289.1653756483".
the Y of the a60r1 is "1480861.2658517451".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "545168.092671901".
the Y of the a62r1 is "1480859.8065726862".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "546954.6098104208".
the Y of the a65r1 is "1480781.8853797023".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "546228.8569621935".
the Y of the a66r1 is "1480818.4505890652".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "546211.67916434".
the Y of the a68r1 is "1479754.503121267".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "546541.7599936102".
the Y of the a69r1 is "1480700.7410306528".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "546914.9971948998".
the Y of the a70r1 is "1480750.7717395264".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "545531.1030054239".
the Y of the a72r1 is "1480730.557038129".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "545531.7860830324".
the Y of the a73r1 is "1479738.5".

[create a4r2]
a4r2 is a area. "An area (4) in r2".
the printed name of the a4r2 is "Room 2".
Understand "Area 4 in Room 2" as a4r2.
the parent of the a4r2 is r2.
the X of the a4r2 is "545525.0".
the Y of the a4r2 is "1478968.5778301887".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "545891.7817693063".
the Y of the a5r2 is "1478844.9748289343".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "545143.389199255".
the Y of the a6r2 is "1478840.2597765364".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "545949.0".
the Y of the a25r2 is "1478880.0".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "545906.1485252035".
the Y of the a26r2 is "1478861.8557671143".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "545098.0".
the Y of the a45r2 is "1478791.0".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "545129.2137161086".
the Y of the a46r2 is "1478816.633971292".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "545089.0".
the Y of the a47r2 is "1478880.0".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "545120.0059757694".
the Y of the a48r2 is "1478860.657907662".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "545940.0".
the Y of the a49r2 is "1478791.0".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "545908.7862838916".
the Y of the a50r2 is "1478816.6339712918".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547296.8816727516".
the Y of the a7r3 is "1478819.4952958587".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "547400.0086022716".
the Y of the a8r3 is "1478964.0104014394".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547413.7496886393".
the Y of the a11r3 is "1478962.317373606".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547507.2968536066".
the Y of the a12r3 is "1478851.4209743899".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "547266.0".
the Y of the a52r3 is "1478772.0".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "547548.0".
the Y of the a54r3 is "1478800.0".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "548154.0".
the Y of the a37r4 is "1480904.0".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "548120.6666666666".
the Y of the a38r4 is "1480853.536449501".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "547954.0".
the Y of the a57r4 is "1480904.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "547987.3333333331".
the Y of the a58r4 is "1480860.9694444446".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548054.5167685037".
the Y of the a61r4 is "1480773.5133957593".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "548059.8708448356".
the Y of the a74r4 is "1480605.3162913683".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "548558.7043713275".
the Y of the a9r5 is "1478822.706711225".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "548654.2390740267".
the Y of the a10r5 is "1478962.431282769".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "548528.0".
the Y of the a53r5 is "1478772.0".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "548811.0".
the Y of the a55r5 is "1478800.0".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "548770.4539705468".
the Y of the a56r5 is "1478848.7643334027".

[create a67r5]
a67r5 is a area. "An area (67) in r5".
the printed name of the a67r5 is "Room 5".
Understand "Area 67 in Room 5" as a67r5.
the parent of the a67r5 is r5.
the X of the a67r5 is "548676.2896516463".
the Y of the a67r5 is "1478960.8833195835".

[create a15r6]
a15r6 is a area. "An area (15) in r6".
the printed name of the a15r6 is "Room 6".
Understand "Area 15 in Room 6" as a15r6.
the parent of the a15r6 is r6.
the X of the a15r6 is "548805.0".
the Y of the a15r6 is "1480914.0".

[create a16r6]
a16r6 is a area. "An area (16) in r6".
the printed name of the a16r6 is "Room 6".
Understand "Area 16 in Room 6" as a16r6.
the parent of the a16r6 is r6.
the X of the a16r6 is "548835.9153780069".
the Y of the a16r6 is "1480888.1834621993".

[create a17r6]
a17r6 is a area. "An area (17) in r6".
the printed name of the a17r6 is "Room 6".
Understand "Area 17 in Room 6" as a17r6.
the parent of the a17r6 is r6.
the X of the a17r6 is "549564.0".
the Y of the a17r6 is "1480914.0".

[create a18r6]
a18r6 is a area. "An area (18) in r6".
the printed name of the a18r6 is "Room 6".
Understand "Area 18 in Room 6" as a18r6.
the parent of the a18r6 is r6.
the X of the a18r6 is "549532.9123931624".
the Y of the a18r6 is "1480888.1662393163".

[create a31r6]
a31r6 is a area. "An area (31) in r6".
the printed name of the a31r6 is "Room 6".
Understand "Area 31 in Room 6" as a31r6.
the parent of the a31r6 is r6.
the X of the a31r6 is "548796.0".
the Y of the a31r6 is "1480824.0".

[create a32r6]
a32r6 is a area. "An area (32) in r6".
the printed name of the a32r6 is "Room 6".
Understand "Area 32 in Room 6" as a32r6.
the parent of the a32r6 is r6.
the X of the a32r6 is "548836.7622699386".
the Y of the a32r6 is "1480842.6487730062".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "549573.0".
the Y of the a33r6 is "1480824.0".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "549532.662577747".
the Y of the a34r6 is "1480842.6912577746".

[create a63r6]
a63r6 is a area. "An area (63) in r6".
the printed name of the a63r6 is "Room 6".
Understand "Area 63 in Room 6" as a63r6.
the parent of the a63r6 is r6.
the X of the a63r6 is "548852.4921500897".
the Y of the a63r6 is "1480860.7986347184".

[create a64r6]
a64r6 is a area. "An area (64) in r6".
the printed name of the a64r6 is "Room 6".
Understand "Area 64 in Room 6" as a64r6.
the parent of the a64r6 is r6.
the X of the a64r6 is "549516.5509290386".
the Y of the a64r6 is "1480861.0062088708".

[create a71r6]
a71r6 is a area. "An area (71) in r6".
the printed name of the a71r6 is "Room 6".
Understand "Area 71 in Room 6" as a71r6.
the parent of the a71r6 is r6.
the X of the a71r6 is "549186.9486065779".
the Y of the a71r6 is "1480778.1064398694".

[create d0]
d0 is a door. "A door between a1r0 and a2r1".
d0 is north of a1r0 and south of a2r1.

[create d2]
d2 is a door. "A door between a67r5 and a71r6".
d2 is north of a67r5 and south of a71r6.

[create d4]
d4 is a door. "A door between a11r3 and a74r4".
d4 is north of a11r3 and south of a74r4.

[create d6]
d6 is a door. "A door between a70r1 and a8r3".
d6 is south of a70r1 and north of a8r3.

[create d8]
d8 is a door. "A door between a74r4 and a10r5".
d8 is southeast of a74r4 and northwest of a10r5.

[create d10]
d10 is a door. "A door between a73r1 and a4r2".
d10 is south of a73r1 and north of a4r2.

east of a0r0 is west of a1r0.

northwest of a0r0 is southeast of a3r0.

east of a4r2 is west of a5r2.

west of a4r2 is east of a6r2.

northeast of a7r3 is southwest of a8r3.

northeast of a9r5 is southwest of a10r5.

southeast of a11r3 is northwest of a12r3.

southeast of a13r1 is northwest of a14r1.

southeast of a15r6 is northwest of a16r6.

southwest of a17r6 is northeast of a18r6.

west of a19r1 is east of a20r1.

east of a21r1 is west of a22r1.

east of a23r1 is west of a24r1.

southwest of a25r2 is northeast of a26r2.

north of a27r1 is south of a28r1.

northeast of a29r1 is southwest of a30r1.

northeast of a31r6 is southwest of a32r6.

northwest of a33r6 is southeast of a34r6.

northwest of a35r1 is southeast of a36r1.

southwest of a37r4 is northeast of a38r4.

southwest of a39r1 is northeast of a40r1.

north of a41r0 is south of a42r0.

north of a43r0 is south of a44r0.

northeast of a45r2 is southwest of a46r2.

southeast of a47r2 is northwest of a48r2.

northwest of a49r2 is southeast of a50r2.

north of a3r0 is south of a51r0.

southwest of a7r3 is northeast of a52r3.

southwest of a9r5 is northeast of a53r5.

southeast of a12r3 is northwest of a54r3.

northwest of a55r5 is southeast of a56r5.

southeast of a57r4 is northwest of a58r4.

south of a24r1 is north of a59r1.

northwest of a36r1 is southeast of a60r1.

southwest of a38r4 is northeast of a61r4.

southwest of a40r1 is northeast of a60r1.

southeast of a14r1 is northwest of a62r1.

southeast of a16r6 is northwest of a63r6.

southwest of a18r6 is northeast of a64r6.

south of a20r1 is north of a65r1.

south of a22r1 is north of a66r1.

east of a10r5 is west of a67r5.

east of a8r3 is west of a11r3.

north of a28r1 is south of a68r1.

southwest of a59r1 is northeast of a69r1.

southwest of a65r1 is northeast of a70r1.

northeast of a30r1 is southwest of a62r1.

northeast of a32r6 is southwest of a63r6.

northwest of a34r6 is southeast of a64r6.

southwest of a0r0 is northeast of a42r0.

south of a1r0 is north of a44r0.

southwest of a6r2 is northeast of a46r2.

northwest of a6r2 is southeast of a48r2.

southeast of a5r2 is northwest of a50r2.

northeast of a5r2 is southwest of a26r2.

east of a63r6 is west of a71r6.

west of a60r1 is east of a65r1.

west of a64r6 is east of a71r6.

south of a72r1 is north of a73r1.

north of a2r1 is south of a69r1.

south of a61r4 is north of a74r4.

west of a66r1 is east of a72r1.

east of a62r1 is west of a72r1.

east of a66r1 is west of a69r1.

east of a59r1 is west of a70r1.

southeast of a58r4 is northwest of a61r4.

northwest of a56r5 is southeast of a67r5.

west of a68r1 is east of a73r1.

southwest of a2r1 is northeast of a68r1.

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
the player is in a2r1.

the orientation of the player is 6.
