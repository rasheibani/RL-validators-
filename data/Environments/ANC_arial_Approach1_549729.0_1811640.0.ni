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

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "545925.8552248292".
the Y of the a5r0 is "1811515.7111410182".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546067.3115300582".
the Y of the a6r0 is "1811511.7077453677".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "545872.7021540478".
the Y of the a14r0 is "1811393.1282482506".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "546154.0".
the Y of the a40r0 is "1811640.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "546114.7500000001".
the Y of the a41r0 is "1811582.1086522108".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "545840.0".
the Y of the a42r0 is "1811640.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "545879.2499999999".
the Y of the a43r0 is "1811583.9232970907".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "545990.0".
the Y of the a58r0 is "1811409.0".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "545979.8896296294".
the Y of the a59r0 is "1811372.757777778".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "546800.2129628072".
the Y of the a18r1 is "1809873.3306140339".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "545482.4898075783".
the Y of the a20r1 is "1810212.9285673478".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "546535.6909520114".
the Y of the a21r1 is "1810219.009107004".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546852.3992464705".
the Y of the a25r1 is "1809592.2474384871".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547049.5".
the Y of the a29r1 is "1809411.1999999997".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "545033.75".
the Y of the a30r1 is "1809426.6525957664".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "545204.0980554341".
the Y of the a31r1 is "1809544.5162008426".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "546744.0".
the Y of the a32r1 is "1809450.5799457994".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545305.0".
the Y of the a44r1 is "1809400.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545266.25".
the Y of the a45r1 is "1809455.5991277103".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "547054.0".
the Y of the a67r1 is "1809400.0".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "544995.0".
the Y of the a68r1 is "1809400.0".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "546766.0".
the Y of the a69r1 is "1809400.0".

[create a0r2]
a0r2 is a area. "An area (0) in r2".
the printed name of the a0r2 is "Room 2".
Understand "Area 0 in Room 2" as a0r2.
the parent of the a0r2 is r2.
the X of the a0r2 is "546803.3749999999".
the Y of the a0r2 is "1811601.9200000002".

[create a1r2]
a1r2 is a area. "An area (1) in r2".
the printed name of the a1r2 is "Room 2".
Understand "Area 1 in Room 2" as a1r2.
the parent of the a1r2 is r2.
the X of the a1r2 is "546959.4818311565".
the Y of the a1r2 is "1811445.9093811852".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "546982.2145260032".
the Y of the a15r2 is "1811369.7337908011".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "547065.0".
the Y of the a38r2 is "1811640.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547027.6250000002".
the Y of the a39r2 is "1811568.840200135".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "546766.0".
the Y of the a64r2 is "1811640.0".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "550122.6635692202".
the Y of the a16r3 is "1809987.5170563739".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "550260.23355668".
the Y of the a17r3 is "1810070.6741780003".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "550042.5623291695".
the Y of the a19r3 is "1810127.7313395664".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "548456.6445220402".
the Y of the a22r3 is "1810140.311749852".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "548281.28175408".
the Y of the a23r3 is "1809671.3480492607".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "548304.9971649275".
the Y of the a24r3 is "1809593.0780053898".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "548361.598773634".
the Y of the a26r3 is "1809869.4665849174".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "548707.3626365624".
the Y of the a27r3 is "1809777.3438961569".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "550084.6555996798".
the Y of the a28r3 is "1809842.8552974607".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "548461.625".
the Y of the a33r3 is "1809436.1128142246".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "549079.6022888336".
the Y of the a34r3 is "1809542.1318872178".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "549692.9665046118".
the Y of the a35r3 is "1809538.7878822302".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "548200.0".
the Y of the a46r3 is "1809400.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548237.375".
the Y of the a47r3 is "1809471.1563946758".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "548594.0".
the Y of the a48r3 is "1809685.0".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "548684.8043567985".
the Y of the a49r3 is "1809756.7947726452".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "550214.0".
the Y of the a50r3 is "1809756.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "550121.9477917004".
the Y of the a51r3 is "1809817.0529526705".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "548499.0".
the Y of the a52r3 is "1809905.5793991417".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "548491.0451689765".
the Y of the a53r3 is "1809869.4665849172".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "549738.0".
the Y of the a60r3 is "1809400.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "549703.7".
the Y of the a61r3 is "1809507.8898876405".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "549039.0".
the Y of the a62r3 is "1809400.0".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "549074.6".
the Y of the a63r3 is "1809525.5734335838".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "548499.0".
the Y of the a70r3 is "1809400.0".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "550305.0".
the Y of the a71r3 is "1810095.0".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "550014.0".
the Y of the a72r3 is "1810168.0".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "548254.8750000002".
the Y of the a2r4 is "1811601.7608695652".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "548359.211053879".
the Y of the a3r4 is "1811500.171599457".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "548464.1250000002".
the Y of the a4r4 is "1811604.2093970669".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "548459.2332245505".
the Y of the a7r4 is "1810901.560350113".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "550009.1472802769".
the Y of the a8r4 is "1810977.4773217994".

[create a9r4]
a9r4 is a area. "An area (9) in r4".
the printed name of the a9r4 is "Room 4".
Understand "Area 9 in Room 4" as a9r4.
the parent of the a9r4 is r4.
the X of the a9r4 is "550080.1924349966".
the Y of the a9r4 is "1811112.1997239746".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "550223.6090407216".
the Y of the a10r4 is "1811031.103810641".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "550050.2356620994".
the Y of the a11r4 is "1811214.3319104237".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "549684.5314170674".
the Y of the a12r4 is "1811503.189299077".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "548731.0925259136".
the Y of the a13r4 is "1811275.7851367455".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "549729.0".
the Y of the a36r4 is "1811640.0".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "549696.0000000002".
the Y of the a37r4 is "1811536.7709580841".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "550179.0".
the Y of the a54r4 is "1811306.0".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "550084.1699148412".
the Y of the a55r4 is "1811237.0028792769".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "548624.0".
the Y of the a56r4 is "1811378.0".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "548715.0800289464".
the Y of the a57r4 is "1811292.3243521973".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "548220.0".
the Y of the a65r4 is "1811640.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548499.0".
the Y of the a66r4 is "1811640.0".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "549981.0".
the Y of the a73r4 is "1810936.0".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "550268.0".
the Y of the a74r4 is "1811003.0".

[create d0]
d0 is a door. "A door between a15r2 and a23r3".
d0 is southeast of a15r2 and northwest of a23r3.

[create d2]
d2 is a door. "A door between a18r1 and a15r2".
d2 is north of a18r1 and south of a15r2.

[create d4]
d4 is a door. "A door between a22r3 and a7r4".
d4 is north of a22r3 and south of a7r4.

[create d6]
d6 is a door. "A door between a14r0 and a20r1".
d6 is south of a14r0 and north of a20r1.

[create d7]
d7 is a door. "A door between a6r0 and a21r1".
d7 is south of a6r0 and north of a21r1.

southeast of a0r2 is northwest of a1r2.

southeast of a2r4 is northwest of a3r4.

northeast of a3r4 is southwest of a4r4.

east of a5r0 is west of a6r0.

south of a3r4 is north of a7r4.

northeast of a8r4 is southwest of a9r4.

southeast of a9r4 is northwest of a10r4.

north of a9r4 is south of a11r4.

northwest of a11r4 is southeast of a12r4.

west of a12r4 is east of a13r4.

southwest of a5r0 is northeast of a14r0.

south of a1r2 is north of a15r2.

northeast of a16r3 is southwest of a17r3.

northwest of a16r3 is southeast of a19r3.

east of a20r1 is west of a21r1.

northeast of a7r4 is southwest of a13r4.

south of a23r3 is north of a24r3.

south of a18r1 is north of a25r1.

north of a23r3 is south of a26r3.

southeast of a22r3 is northwest of a27r3.

south of a16r3 is north of a28r3.

south of a22r3 is north of a26r3.

northwest of a18r1 is southeast of a21r1.

southeast of a25r1 is northwest of a29r1.

northeast of a30r1 is southwest of a31r1.

southwest of a25r1 is northeast of a32r1.

southeast of a24r3 is northwest of a33r3.

east of a34r3 is west of a35r3.

southeast of a27r3 is northwest of a34r3.

southwest of a20r1 is northeast of a31r1.

southwest of a28r3 is northeast of a35r3.

south of a36r4 is north of a37r4.

southwest of a38r2 is northeast of a39r2.

southwest of a40r0 is northeast of a41r0.

southeast of a42r0 is northwest of a43r0.

northwest of a44r1 is southeast of a45r1.

northeast of a46r3 is southwest of a47r3.

northeast of a48r3 is southwest of a49r3.

northwest of a50r3 is southeast of a51r3.

south of a52r3 is north of a53r3.

southwest of a54r4 is northeast of a55r4.

southeast of a56r4 is northwest of a57r4.

south of a58r0 is north of a59r0.

north of a60r3 is south of a61r3.

north of a62r3 is south of a63r3.

northwest of a0r2 is southeast of a64r2.

northwest of a2r4 is southeast of a65r4.

northeast of a4r4 is southwest of a66r4.

south of a29r1 is north of a67r1.

southwest of a30r1 is northeast of a68r1.

southeast of a32r1 is northwest of a69r1.

southeast of a33r3 is northwest of a70r3.

northeast of a17r3 is southwest of a71r3.

northwest of a19r3 is southeast of a72r3.

southwest of a8r4 is northeast of a73r4.

southeast of a10r4 is northwest of a74r4.

south of a35r3 is north of a61r3.

south of a34r3 is north of a63r3.

southeast of a31r1 is northwest of a45r1.

southwest of a24r3 is northeast of a47r3.

southwest of a27r3 is northeast of a49r3.

southeast of a28r3 is northwest of a51r3.

east of a26r3 is west of a53r3.

northeast of a11r4 is southwest of a55r4.

northwest of a13r4 is southeast of a57r4.

east of a14r0 is west of a59r0.

north of a12r4 is south of a37r4.

northeast of a1r2 is southwest of a39r2.

northeast of a6r0 is southwest of a41r0.

northwest of a5r0 is southeast of a43r0.

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
the player is in a36r4.

the orientation of the player is 6.
