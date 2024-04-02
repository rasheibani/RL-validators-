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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "545456.3319092895".
the Y of the a0r0 is "1414329.4378453037".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "545811.822886608".
the Y of the a1r0 is "1414240.061504006".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "545127.3913043479".
the Y of the a6r0 is "1414245.5".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "545866.0".
the Y of the a57r0 is "1414280.0".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "545838.8491587283".
the Y of the a58r0 is "1414260.2680626004".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "545080.0".
the Y of the a63r0 is "1414191.0".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "545110.6227296316".
the Y of the a64r0 is "1414216.5742086142".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "545071.0".
the Y of the a69r0 is "1414280.0".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "545113.2240682533".
the Y of the a70r0 is "1414261.7923215088".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "545857.0".
the Y of the a71r0 is "1414191.0".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "545825.9579422104".
the Y of the a72r0 is "1414216.6166125855".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "545559.025775421".
the Y of the a7r1 is "1416126.6399497383".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "545641.8999999999".
the Y of the a8r1 is "1416248.1052980132".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545582.0735241218".
the Y of the a19r1 is "1416011.726733836".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "545189.1884979301".
the Y of the a21r1 is "1416259.7787601023".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "545141.0".
the Y of the a33r1 is "1416314.0".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "545172.0876068376".
the Y of the a34r1 is "1416288.1662393163".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545132.0".
the Y of the a45r1 is "1416224.0".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "545174.2698675497".
the Y of the a46r1 is "1416242.4980132452".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "545676.0".
the Y of the a74r1 is "1416304.0".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "546292.5661679945".
the Y of the a24r2 is "1414219.9341592467".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "546355.0091290365".
the Y of the a25r2 is "1414330.1696830266".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "546357.7606867261".
the Y of the a26r2 is "1414349.5866818968".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "546459.0".
the Y of the a41r2 is "1414203.0".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "546415.8503889912".
the Y of the a42r2 is "1414261.5027281872".

[create a73r2]
a73r2 is a area. "An area (73) in r2".
the printed name of the a73r2 is "Room 2".
Understand "Area 73 in Room 2" as a73r2.
the parent of the a73r2 is r2.
the X of the a73r2 is "546269.0".
the Y of the a73r2 is "1414169.0".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547282.5378625151".
the Y of the a9r3 is "1416087.8599739138".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547470.6012042759".
the Y of the a10r3 is "1416113.3926818105".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547643.5946109599".
the Y of the a11r3 is "1416242.59858462".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547879.9151692868".
the Y of the a12r3 is "1416261.6108476832".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547569.3847023154".
the Y of the a17r3 is "1415961.9719077216".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "547234.013904575".
the Y of the a18r3 is "1415995.0875486657".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "547131.8892105871".
the Y of the a20r3 is "1416261.7946907768".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "547085.0".
the Y of the a31r3 is "1416314.0".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "547115.9153780068".
the Y of the a32r3 is "1416288.1834621993".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547927.0".
the Y of the a35r3 is "1416314.0".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "547895.9123931623".
the Y of the a36r3 is "1416288.1662393163".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "547683.0".
the Y of the a37r3 is "1416304.0".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547650.0059829061".
the Y of the a38r3 is "1416305.7505982905".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "547427.0".
the Y of the a43r3 is "1415927.0252951097".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "547448.798894772".
the Y of the a44r3 is "1415896.1877459248".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "547076.0".
the Y of the a47r3 is "1416224.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "547094.5729961733".
the Y of the a48r3 is "1416244.8677003826".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "547936.0".
the Y of the a49r3 is "1416224.0".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "547896.9448424069".
the Y of the a50r3 is "1416242.8194842406".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "548071.0".
the Y of the a2r4 is "1414368.5778301887".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "548437.7290448345".
the Y of the a3r4 is "1414245.062703054".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "547338.0668298496".
the Y of the a4r4 is "1414365.8958366967".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "547672.3376146692".
the Y of the a5r4 is "1414243.3280303988".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "548066.1730804583".
the Y of the a16r4 is "1415059.7005024815".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "547720.8667805495".
the Y of the a22r4 is "1414242.6549881685".

[create a23r4]
a23r4 is a area. "An area (23) in r4".
the printed name of the a23r4 is "Room 4".
Understand "Area 23 in Room 4" as a23r4.
the parent of the a23r4 is r4.
the X of the a23r4 is "546952.9132133439".
the Y of the a23r4 is "1414240.22214224".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "548495.0".
the Y of the a39r4 is "1414280.0".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "548451.9383292026".
the Y of the a40r4 is "1414261.8770228897".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "546908.0".
the Y of the a55r4 is "1414191.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "546939.0234020578".
the Y of the a56r4 is "1414217.0166743887".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "547696.6356589147".
the Y of the a59r4 is "1414193.441860465".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "547723.0126693366".
the Y of the a60r4 is "1414196.3991632005".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "546898.0".
the Y of the a61r4 is "1414280.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "546929.3505784649".
the Y of the a62r4 is "1414260.9916380297".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "548486.0".
the Y of the a65r4 is "1414191.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548454.7862838916".
the Y of the a66r4 is "1414216.6339712918".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "547644.0".
the Y of the a67r4 is "1414191.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "547670.1365753084".
the Y of the a68r4 is "1414196.1279641096".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "548357.7072666287".
the Y of the a13r5 is "1416261.4777749956".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "548701.3793965107".
the Y of the a14r5 is "1416174.0644405335".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "549022.3051002845".
the Y of the a15r5 is "1416261.4983535397".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "548311.0".
the Y of the a27r5 is "1416314.0".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "548341.8949627934".
the Y of the a28r5 is "1416287.7894485786".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "549069.0".
the Y of the a29r5 is "1416314.0".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "549038.1050372066".
the Y of the a30r5 is "1416287.7894485786".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "548301.0".
the Y of the a51r5 is "1416224.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "548339.7029286582".
the Y of the a52r5 is "1416242.4774523713".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "549079.0".
the Y of the a53r5 is "1416224.0".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "549039.9982014389".
the Y of the a54r5 is "1416242.4442446043".

[create d0]
d0 is a door. "A door between a0r0 and a19r1".
d0 is north of a0r0 and south of a19r1.

[create d2]
d2 is a door. "A door between a26r2 and a18r3".
d2 is northeast of a26r2 and southwest of a18r3.

[create d4]
d4 is a door. "A door between a19r1 and a26r2".
d4 is southeast of a19r1 and northwest of a26r2.

[create d6]
d6 is a door. "A door between a17r3 and a16r4".
d6 is southeast of a17r3 and northwest of a16r4.

[create d7]
d7 is a door. "A door between a18r3 and a4r4".
d7 is south of a18r3 and north of a4r4.

[create d8]
d8 is a door. "A door between a16r4 and a14r5".
d8 is northeast of a16r4 and southwest of a14r5.

east of a0r0 is west of a1r0.

east of a2r4 is west of a3r4.

east of a4r4 is west of a5r4.

west of a0r0 is east of a6r0.

northeast of a7r1 is southwest of a8r1.

east of a9r3 is west of a10r3.

east of a11r3 is west of a12r3.

east of a13r5 is west of a14r5.

east of a14r5 is west of a15r5.

southeast of a10r3 is northwest of a17r3.

southwest of a9r3 is northeast of a18r3.

south of a7r1 is north of a19r1.

northwest of a9r3 is southeast of a20r3.

northeast of a10r3 is southwest of a11r3.

west of a7r1 is east of a21r1.

west of a2r4 is east of a22r4.

west of a4r4 is east of a23r4.

northeast of a24r2 is southwest of a25r2.

north of a2r4 is south of a16r4.

southeast of a27r5 is northwest of a28r5.

southwest of a29r5 is northeast of a30r5.

southeast of a31r3 is northwest of a32r3.

southeast of a33r1 is northwest of a34r1.

southwest of a35r3 is northeast of a36r3.

west of a37r3 is east of a38r3.

southwest of a39r4 is northeast of a40r4.

northwest of a41r2 is southeast of a42r2.

southeast of a43r3 is northwest of a44r3.

northeast of a45r1 is southwest of a46r1.

northeast of a47r3 is southwest of a48r3.

northwest of a49r3 is southeast of a50r3.

northeast of a51r5 is southwest of a52r5.

northwest of a53r5 is southeast of a54r5.

northeast of a55r4 is southwest of a56r4.

southwest of a57r0 is northeast of a58r0.

east of a59r4 is west of a60r4.

southeast of a61r4 is northwest of a62r4.

northeast of a63r0 is southwest of a64r0.

northwest of a65r4 is southeast of a66r4.

east of a67r4 is west of a68r4.

southeast of a69r0 is northwest of a70r0.

northwest of a71r0 is southeast of a72r0.

southwest of a24r2 is northeast of a73r2.

northeast of a8r1 is southwest of a74r1.

southwest of a21r1 is northeast of a46r1.

southwest of a20r3 is northeast of a48r3.

southeast of a12r3 is northwest of a50r3.

southwest of a13r5 is northeast of a52r5.

southeast of a15r5 is northwest of a54r5.

northwest of a13r5 is southeast of a28r5.

northeast of a15r5 is southwest of a30r5.

northwest of a20r3 is southeast of a32r3.

northwest of a21r1 is southeast of a34r1.

southeast of a3r4 is northwest of a66r4.

south of a5r4 is north of a68r4.

east of a5r4 is west of a22r4.

northwest of a6r0 is southeast of a70r0.

northeast of a3r4 is southwest of a40r4.

southeast of a25r2 is northwest of a42r2.

north of a25r2 is south of a26r2.

southwest of a17r3 is northeast of a44r3.

southeast of a1r0 is northwest of a72r0.

southwest of a23r4 is northeast of a56r4.

northeast of a1r0 is southwest of a58r0.

south of a22r4 is north of a60r4.

northwest of a23r4 is southeast of a62r4.

southwest of a6r0 is northeast of a64r0.

northeast of a12r3 is southwest of a36r3.

north of a11r3 is south of a38r3.

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
the player is in a62r4.

the orientation of the player is 6.
