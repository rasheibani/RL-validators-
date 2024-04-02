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

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "547302.8224123432".
the Y of the a8r0 is "1748620.0921328291".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "547405.9861188938".
the Y of the a9r0 is "1748764.0131715974".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "547419.7496886395".
the Y of the a33r0 is "1748762.3173736061".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "547513.2968536066".
the Y of the a34r0 is "1748651.4209743899".

[create a97r0]
a97r0 is a area. "An area (97) in r0".
the printed name of the a97r0 is "Room 0".
Understand "Area 97 in Room 0" as a97r0.
the parent of the a97r0 is r0.
the X of the a97r0 is "547272.0".
the Y of the a97r0 is "1748572.0".

[create a99r0]
a99r0 is a area. "An area (99) in r0".
the printed name of the a99r0 is "Room 0".
Understand "Area 99 in Room 0" as a99r0.
the parent of the a99r0 is r0.
the X of the a99r0 is "547554.0".
the Y of the a99r0 is "1748600.0".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "548060.5167685037".
the Y of the a26r1 is "1750573.513395759".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "548065.8708448355".
the Y of the a27r1 is "1750405.316291368".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547993.3333333331".
the Y of the a32r1 is "1750660.9694444442".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "548160.0".
the Y of the a55r1 is "1750704.0".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "548126.6666666667".
the Y of the a56r1 is "1750653.5364495006".

[create a96r1]
a96r1 is a area. "An area (96) in r1".
the printed name of the a96r1 is "Room 1".
Understand "Area 96 in Room 1" as a96r1.
the parent of the a96r1 is r1.
the X of the a96r1 is "547960.0".
the Y of the a96r1 is "1750704.0".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "548564.7043713275".
the Y of the a10r2 is "1748622.7067112252".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "548660.2390740265".
the Y of the a11r2 is "1748762.431282769".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "548682.2896516463".
the Y of the a35r2 is "1748760.8833195833".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "548776.4539705466".
the Y of the a36r2 is "1748648.7643334025".

[create a95r2]
a95r2 is a area. "An area (95) in r2".
the printed name of the a95r2 is "Room 2".
Understand "Area 95 in Room 2" as a95r2.
the parent of the a95r2 is r2.
the X of the a95r2 is "548817.0".
the Y of the a95r2 is "1748600.0".

[create a98r2]
a98r2 is a area. "An area (98) in r2".
the printed name of the a98r2 is "Room 2".
Understand "Area 98 in Room 2" as a98r2.
the parent of the a98r2 is r2.
the X of the a98r2 is "548534.0".
the Y of the a98r2 is "1748572.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "549193.1587808632".
the Y of the a19r3 is "1750581.3772771838".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "549510.1402545043".
the Y of the a20r3 is "1750541.7600531357".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "549522.7207755846".
the Y of the a21r3 is "1750555.191306174".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "549867.8606168444".
the Y of the a22r3 is "1750659.860290629".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "548858.4921500895".
the Y of the a30r3 is "1750660.7986347184".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "549146.4944995969".
the Y of the a31r3 is "1750611.0205706113".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "549916.0".
the Y of the a39r3 is "1750714.0".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "549884.9123931623".
the Y of the a40r3 is "1750688.1662393163".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "549105.0".
the Y of the a41r3 is "1750714.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "549136.8264604809".
the Y of the a42r3 is "1750705.767353952".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "549571.0".
the Y of the a47r3 is "1750704.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "549538.0059829059".
the Y of the a48r3 is "1750705.7505982905".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "548811.0".
the Y of the a61r3 is "1750714.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "548841.9153780066".
the Y of the a62r3 is "1750688.1834621995".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "549925.0".
the Y of the a75r3 is "1750624.0".

[create a76r3]
a76r3 is a area. "An area (76) in r3".
the printed name of the a76r3 is "Room 3".
Understand "Area 76 in Room 3" as a76r3.
the parent of the a76r3 is r3.
the X of the a76r3 is "549882.9400000001".
the Y of the a76r3 is "1750642.519".

[create a77r3]
a77r3 is a area. "An area (77) in r3".
the printed name of the a77r3 is "Room 3".
Understand "Area 77 in Room 3" as a77r3.
the parent of the a77r3 is r3.
the X of the a77r3 is "548802.0".
the Y of the a77r3 is "1750624.0".

[create a78r3]
a78r3 is a area. "An area (78) in r3".
the printed name of the a78r3 is "Room 3".
Understand "Area 78 in Room 3" as a78r3.
the parent of the a78r3 is r3.
the X of the a78r3 is "548842.7622699386".
the Y of the a78r3 is "1750642.6487730062".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "549128.3891992549".
the Y of the a0r4 is "1748640.2597765364".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "549510.0".
the Y of the a1r4 is "1748768.5778301887".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "549876.7817693062".
the Y of the a13r4 is "1748644.9748289345".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "549934.0".
the Y of the a73r4 is "1748680.0".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "549891.1485252036".
the Y of the a74r4 is "1748661.855767114".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "549083.0".
the Y of the a81r4 is "1748591.0".

[create a82r4]
a82r4 is a area. "An area (82) in r4".
the printed name of the a82r4 is "Room 4".
Understand "Area 82 in Room 4" as a82r4.
the parent of the a82r4 is r4.
the X of the a82r4 is "549114.2137161085".
the Y of the a82r4 is "1748616.633971292".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "549074.0".
the Y of the a89r4 is "1748680.0".

[create a90r4]
a90r4 is a area. "An area (90) in r4".
the printed name of the a90r4 is "Room 4".
Understand "Area 90 in Room 4" as a90r4.
the parent of the a90r4 is r4.
the X of the a90r4 is "549105.0059757694".
the Y of the a90r4 is "1748660.657907662".

[create a93r4]
a93r4 is a area. "An area (93) in r4".
the printed name of the a93r4 is "Room 4".
Understand "Area 93 in Room 4" as a93r4.
the parent of the a93r4 is r4.
the X of the a93r4 is "549925.0".
the Y of the a93r4 is "1748591.0".

[create a94r4]
a94r4 is a area. "An area (94) in r4".
the printed name of the a94r4 is "Room 4".
Understand "Area 94 in Room 4" as a94r4.
the parent of the a94r4 is r4.
the X of the a94r4 is "549893.7862838916".
the Y of the a94r4 is "1748616.6339712918".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "546093.0873366152".
the Y of the a12r5 is "1749649.159650314".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "546358.9589252132".
the Y of the a14r5 is "1750661.6795222145".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "546557.3701298707".
the Y of the a15r5 is "1750642.9458874462".

[create a16r5]
a16r5 is a area. "An area (16) in r5".
the printed name of the a16r5 is "Room 5".
Understand "Area 16 in Room 5" as a16r5.
the parent of the a16r5 is r5.
the X of the a16r5 is "546916.3787096874".
the Y of the a16r5 is "1750543.7282370713".

[create a17r5]
a17r5 is a area. "An area (17) in r5".
the printed name of the a17r5 is "Room 5".
Understand "Area 17 in Room 5" as a17r5.
the parent of the a17r5 is r5.
the X of the a17r5 is "547029.8869813068".
the Y of the a17r5 is "1750615.9284325424".

[create a18r5]
a18r5 is a area. "An area (18) in r5".
the printed name of the a18r5 is "Room 5".
Understand "Area 18 in Room 5" as a18r5.
the parent of the a18r5 is r5.
the X of the a18r5 is "546711.3619459791".
the Y of the a18r5 is "1750565.4949057791".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "547295.1653756483".
the Y of the a23r5 is "1750661.2658517451".

[create a24r5]
a24r5 is a area. "An area (24) in r5".
the printed name of the a24r5 is "Room 5".
Understand "Area 24 in Room 5" as a24r5.
the parent of the a24r5 is r5.
the X of the a24r5 is "546080.6892610956".
the Y of the a24r5 is "1749710.936267989".

[create a25r5]
a25r5 is a area. "An area (25) in r5".
the printed name of the a25r5 is "Room 5".
Understand "Area 25 in Room 5" as a25r5.
the parent of the a25r5 is r5.
the X of the a25r5 is "545503.5427972081".
the Y of the a25r5 is "1750538.7985219036".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545883.9343690767".
the Y of the a28r5 is "1750661.6427193338".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545135.4913012546".
the Y of the a29r5 is "1750662.596106584".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "545931.0".
the Y of the a37r5 is "1750714.0".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "545899.9123931623".
the Y of the a38r5 is "1750688.1662393163".

[create a43r5]
a43r5 is a area. "An area (43) in r5".
the printed name of the a43r5 is "Room 5".
Understand "Area 43 in Room 5" as a43r5.
the parent of the a43r5 is r5.
the X of the a43r5 is "546519.0".
the Y of the a43r5 is "1750714.0".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "546550.9940170939".
the Y of the a44r5 is "1750705.7505982907".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "547071.0".
the Y of the a45r5 is "1750714.0".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "547039.0059829059".
the Y of the a46r5 is "1750705.7505982907".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "547352.0".
the Y of the a49r5 is "1750624.0".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "547311.6686930091".
the Y of the a50r5 is "1750642.2965214455".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "545940.0".
the Y of the a51r5 is "1750624.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "545901.0502336449".
the Y of the a52r5 is "1750642.8300233646".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "545080.0".
the Y of the a53r5 is "1750624.0".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "545116.9075342466".
the Y of the a54r5 is "1750643.0342465756".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "547342.0".
the Y of the a57r5 is "1750714.0".

[create a58r5]
a58r5 is a area. "An area (58) in r5".
the printed name of the a58r5 is "Room 5".
Understand "Area 58 in Room 5" as a58r5.
the parent of the a58r5 is r5.
the X of the a58r5 is "547311.1050372066".
the Y of the a58r5 is "1750687.7894485786".

[create a59r5]
a59r5 is a area. "An area (59) in r5".
the printed name of the a59r5 is "Room 5".
Understand "Area 59 in Room 5" as a59r5.
the parent of the a59r5 is r5.
the X of the a59r5 is "546312.0".
the Y of the a59r5 is "1750714.0".

[create a60r5]
a60r5 is a area. "An area (60) in r5".
the printed name of the a60r5 is "Room 5".
Understand "Area 60 in Room 5" as a60r5.
the parent of the a60r5 is r5.
the X of the a60r5 is "546342.9153780066".
the Y of the a60r5 is "1750688.1834621995".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "545089.0".
the Y of the a63r5 is "1750714.0".

[create a64r5]
a64r5 is a area. "An area (64) in r5".
the printed name of the a64r5 is "Room 5".
Understand "Area 64 in Room 5" as a64r5.
the parent of the a64r5 is r5.
the X of the a64r5 is "545120.0876068377".
the Y of the a64r5 is "1750688.1662393163".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "546303.0".
the Y of the a79r5 is "1750624.0".

[create a80r5]
a80r5 is a area. "An area (80) in r5".
the printed name of the a80r5 is "Room 5".
Understand "Area 80 in Room 5" as a80r5.
the parent of the a80r5 is r5.
the X of the a80r5 is "546341.6344476744".
the Y of the a80r5 is "1750642.8615552324".

[create a2r6]
a2r6 is a area. "An area (2) in r6".
the printed name of the a2r6 is "Room 6".
Understand "Area 2 in Room 6" as a2r6.
the parent of the a2r6 is r6.
the X of the a2r6 is "546335.0670611437".
the Y of the a2r6 is "1748644.7228796843".

[create a3r6]
a3r6 is a area. "An area (3) in r6".
the printed name of the a3r6 is "Room 6".
Understand "Area 3 in Room 6" as a3r6.
the parent of the a3r6 is r6.
the X of the a3r6 is "546713.3355180037".
the Y of the a3r6 is "1748762.335518004".

[create a4r6]
a4r6 is a area. "An area (4) in r6".
the printed name of the a4r6 is "Room 6".
Understand "Area 4 in Room 6" as a4r6.
the parent of the a4r6 is r6.
the X of the a4r6 is "545394.8633422251".
the Y of the a4r6 is "1748734.3385130367".

[create a5r6]
a5r6 is a area. "An area (5) in r6".
the printed name of the a5r6 is "Room 6".
Understand "Area 5 in Room 6" as a5r6.
the parent of the a5r6 is r6.
the X of the a5r6 is "545743.0171235859".
the Y of the a5r6 is "1748642.915905135".

[create a6r6]
a6r6 is a area. "An area (6) in r6".
the printed name of the a6r6 is "Room 6".
Understand "Area 6 in Room 6" as a6r6.
the parent of the a6r6 is r6.
the X of the a6r6 is "547082.7426126123".
the Y of the a6r6 is "1748641.706756757".

[create a7r6]
a7r6 is a area. "An area (7) in r6".
the printed name of the a7r6 is "Room 6".
Understand "Area 7 in Room 6" as a7r6.
the parent of the a7r6 is r6.
the X of the a7r6 is "545077.0941579642".
the Y of the a7r6 is "1748641.5825458067".

[create a65r6]
a65r6 is a area. "An area (65) in r6".
the printed name of the a65r6 is "Room 6".
Understand "Area 65 in Room 6" as a65r6.
the parent of the a65r6 is r6.
the X of the a65r6 is "545022.0".
the Y of the a65r6 is "1748680.0".

[create a66r6]
a66r6 is a area. "An area (66) in r6".
the printed name of the a66r6 is "Room 6".
Understand "Area 66 in Room 6" as a66r6.
the parent of the a66r6 is r6.
the X of the a66r6 is "545058.0183625782".
the Y of the a66r6 is "1748661.1647782382".

[create a67r6]
a67r6 is a area. "An area (67) in r6".
the printed name of the a67r6 is "Room 6".
Understand "Area 67 in Room 6" as a67r6.
the parent of the a67r6 is r6.
the X of the a67r6 is "547138.0".
the Y of the a67r6 is "1748680.0".

[create a68r6]
a68r6 is a area. "An area (68) in r6".
the printed name of the a68r6 is "Room 6".
Understand "Area 68 in Room 6" as a68r6.
the parent of the a68r6 is r6.
the X of the a68r6 is "547101.1452295461".
the Y of the a68r6 is "1748661.24935881".

[create a69r6]
a69r6 is a area. "An area (69) in r6".
the printed name of the a69r6 is "Room 6".
Understand "Area 69 in Room 6" as a69r6.
the parent of the a69r6 is r6.
the X of the a69r6 is "545799.0".
the Y of the a69r6 is "1748680.0".

[create a70r6]
a70r6 is a area. "An area (70) in r6".
the printed name of the a70r6 is "Room 6".
Understand "Area 70 in Room 6" as a70r6.
the parent of the a70r6 is r6.
the X of the a70r6 is "545760.2797213208".
the Y of the a70r6 is "1748661.438005709".

[create a71r6]
a71r6 is a area. "An area (71) in r6".
the printed name of the a71r6 is "Room 6".
Understand "Area 71 in Room 6" as a71r6.
the parent of the a71r6 is r6.
the X of the a71r6 is "546278.0".
the Y of the a71r6 is "1748680.0".

[create a72r6]
a72r6 is a area. "An area (72) in r6".
the printed name of the a72r6 is "Room 6".
Understand "Area 72 in Room 6" as a72r6.
the parent of the a72r6 is r6.
the X of the a72r6 is "546320.2240682533".
the Y of the a72r6 is "1748661.7923215088".

[create a83r6]
a83r6 is a area. "An area (83) in r6".
the printed name of the a83r6 is "Room 6".
Understand "Area 83 in Room 6" as a83r6.
the parent of the a83r6 is r6.
the X of the a83r6 is "545031.0".
the Y of the a83r6 is "1748591.0".

[create a84r6]
a84r6 is a area. "An area (84) in r6".
the printed name of the a84r6 is "Room 6".
Understand "Area 84 in Room 6" as a84r6.
the parent of the a84r6 is r6.
the X of the a84r6 is "545062.0420577896".
the Y of the a84r6 is "1748616.6166125853".

[create a85r6]
a85r6 is a area. "An area (85) in r6".
the printed name of the a85r6 is "Room 6".
Understand "Area 85 in Room 6" as a85r6.
the parent of the a85r6 is r6.
the X of the a85r6 is "547129.0".
the Y of the a85r6 is "1748591.0".

[create a86r6]
a86r6 is a area. "An area (86) in r6".
the printed name of the a86r6 is "Room 6".
Understand "Area 86 in Room 6" as a86r6.
the parent of the a86r6 is r6.
the X of the a86r6 is "547097.7862838916".
the Y of the a86r6 is "1748616.633971292".

[create a87r6]
a87r6 is a area. "An area (87) in r6".
the printed name of the a87r6 is "Room 6".
Understand "Area 87 in Room 6" as a87r6.
the parent of the a87r6 is r6.
the X of the a87r6 is "545790.0".
the Y of the a87r6 is "1748591.0".

[create a88r6]
a88r6 is a area. "An area (88) in r6".
the printed name of the a88r6 is "Room 6".
Understand "Area 88 in Room 6" as a88r6.
the parent of the a88r6 is r6.
the X of the a88r6 is "545758.7862838916".
the Y of the a88r6 is "1748616.633971292".

[create a91r6]
a91r6 is a area. "An area (91) in r6".
the printed name of the a91r6 is "Room 6".
Understand "Area 91 in Room 6" as a91r6.
the parent of the a91r6 is r6.
the X of the a91r6 is "546287.0".
the Y of the a91r6 is "1748591.0".

[create a92r6]
a92r6 is a area. "An area (92) in r6".
the printed name of the a92r6 is "Room 6".
Understand "Area 92 in Room 6" as a92r6.
the parent of the a92r6 is r6.
the X of the a92r6 is "546318.2137161085".
the Y of the a92r6 is "1748616.633971292".

[create d0]
d0 is a door. "A door between a12r5 and a4r6".
d0 is southwest of a12r5 and northeast of a4r6.

[create d1]
d1 is a door. "A door between a12r5 and a3r6".
d1 is southeast of a12r5 and northwest of a3r6.

[create d4]
d4 is a door. "A door between a35r2 and a19r3".
d4 is north of a35r2 and south of a19r3.

[create d6]
d6 is a door. "A door between a33r0 and a27r1".
d6 is north of a33r0 and south of a27r1.

[create d8]
d8 is a door. "A door between a9r0 and a16r5".
d8 is north of a9r0 and south of a16r5.

[create d10]
d10 is a door. "A door between a27r1 and a11r2".
d10 is southeast of a27r1 and northwest of a11r2.

[create d12]
d12 is a door. "A door between a20r3 and a1r4".
d12 is south of a20r3 and north of a1r4.

east of a0r4 is west of a1r4.

east of a2r6 is west of a3r6.

east of a4r6 is west of a5r6.

east of a3r6 is west of a6r6.

west of a4r6 is east of a7r6.

northeast of a8r0 is southwest of a9r0.

northeast of a10r2 is southwest of a11r2.

east of a1r4 is west of a13r4.

east of a14r5 is west of a15r5.

northeast of a16r5 is southwest of a17r5.

southeast of a15r5 is northwest of a18r5.

east of a19r3 is west of a20r3.

east of a21r3 is west of a22r3.

east of a17r5 is west of a23r5.

southwest of a18r5 is northeast of a24r5.

northwest of a24r5 is southeast of a25r5.

south of a26r1 is north of a27r1.

east of a25r5 is west of a28r5.

west of a25r5 is east of a29r5.

west of a16r5 is east of a18r5.

east of a30r3 is west of a31r3.

northwest of a26r1 is southeast of a32r1.

southeast of a33r0 is northwest of a34r0.

southeast of a35r2 is northwest of a36r2.

southwest of a37r5 is northeast of a38r5.

southwest of a39r3 is northeast of a40r3.

east of a41r3 is west of a42r3.

east of a43r5 is west of a44r5.

west of a45r5 is east of a46r5.

west of a47r3 is east of a48r3.

northwest of a49r5 is southeast of a50r5.

northwest of a51r5 is southeast of a52r5.

northeast of a53r5 is southwest of a54r5.

southwest of a55r1 is northeast of a56r1.

southwest of a57r5 is northeast of a58r5.

southeast of a59r5 is northwest of a60r5.

southeast of a61r3 is northwest of a62r3.

southeast of a63r5 is northwest of a64r5.

southeast of a65r6 is northwest of a66r6.

southwest of a67r6 is northeast of a68r6.

southwest of a69r6 is northeast of a70r6.

southeast of a71r6 is northwest of a72r6.

southwest of a73r4 is northeast of a74r4.

northwest of a75r3 is southeast of a76r3.

northeast of a77r3 is southwest of a78r3.

northeast of a79r5 is southwest of a80r5.

northeast of a81r4 is southwest of a82r4.

northeast of a83r6 is southwest of a84r6.

northwest of a85r6 is southeast of a86r6.

northwest of a87r6 is southeast of a88r6.

southeast of a89r4 is northwest of a90r4.

northeast of a91r6 is southwest of a92r6.

northwest of a93r4 is southeast of a94r4.

southeast of a36r2 is northwest of a95r2.

northwest of a32r1 is southeast of a96r1.

southwest of a8r0 is northeast of a97r0.

southwest of a10r2 is northeast of a98r2.

southeast of a34r0 is northwest of a99r0.

northwest of a2r6 is southeast of a72r6.

northeast of a13r4 is southwest of a74r4.

east of a11r2 is west of a35r2.

east of a9r0 is west of a33r0.

north of a12r5 is south of a24r5.

northeast of a20r3 is southwest of a21r3.

northwest of a19r3 is southeast of a31r3.

southeast of a22r3 is northwest of a76r3.

southwest of a30r3 is northeast of a78r3.

southeast of a6r6 is northwest of a86r6.

southeast of a5r6 is northwest of a88r6.

northwest of a0r4 is southeast of a90r4.

southwest of a2r6 is northeast of a92r6.

southeast of a13r4 is northwest of a94r4.

northwest of a7r6 is southeast of a66r6.

northeast of a6r6 is southwest of a68r6.

northeast of a5r6 is southwest of a70r6.

southwest of a0r4 is northeast of a82r4.

southwest of a7r6 is northeast of a84r6.

northwest of a30r3 is southeast of a62r3.

northwest of a29r5 is southeast of a64r5.

northeast of a28r5 is southwest of a38r5.

northeast of a22r3 is southwest of a40r3.

north of a31r3 is south of a42r3.

north of a15r5 is south of a44r5.

north of a17r5 is south of a46r5.

north of a21r3 is south of a48r3.

southwest of a14r5 is northeast of a80r5.

southeast of a23r5 is northwest of a50r5.

southeast of a28r5 is northwest of a52r5.

southwest of a29r5 is northeast of a54r5.

northeast of a26r1 is southwest of a56r1.

northeast of a23r5 is southwest of a58r5.

northwest of a14r5 is southeast of a60r5.

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
the player is in a30r3.

the orientation of the player is 6.
