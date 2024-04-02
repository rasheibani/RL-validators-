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
the X of the a0r0 is "548277.1458986761".
the Y of the a0r0 is "1761942.9609899817".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548643.9394858426".
the Y of the a1r0 is "1762044.021693426".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547961.7161973023".
the Y of the a6r0 is "1762063.6416276314".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "548140.72749368".
the Y of the a7r0 is "1762044.7727607444".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548253.8239941316".
the Y of the a9r0 is "1761948.7162710952".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "549802.5870901747".
the Y of the a10r0 is "1761986.6022325146".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "549896.0550595239".
the Y of the a11r0 is "1762100.0".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "549007.0".
the Y of the a16r0 is "1761937.5322580647".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "549911.5088740578".
the Y of the a23r0 is "1761658.526961484".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "548109.5555637693".
the Y of the a24r0 is "1761658.7341972142".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "548145.2993728241".
the Y of the a25r0 is "1761774.7005733976".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "548683.1111111111".
the Y of the a41r0 is "1762104.0".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "548650.1923615087".
the Y of the a42r0 is "1762105.40408955".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "548095.0".
the Y of the a45r0 is "1762104.824742268".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "548125.9938890934".
the Y of the a46r0 is "1762113.912371134".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "548206.0846797742".
the Y of the a47r0 is "1761793.0167412886".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "548182.2762637076".
the Y of the a48r0 is "1761762.9618778795".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "547908.0".
the Y of the a49r0 is "1762024.0".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "547935.2059471365".
the Y of the a50r0 is "1762044.0044052862".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "547917.0".
the Y of the a59r0 is "1762114.0".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "547945.4275663053".
the Y of the a60r0 is "1762088.4322433695".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "549879.0".
the Y of the a71r0 is "1761588.0".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "549887.2924357145".
the Y of the a72r0 is "1761625.1953469105".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "549962.0".
the Y of the a73r0 is "1761625.0".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "549924.0515730679".
the Y of the a74r0 is "1761654.3302550102".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "548058.0".
the Y of the a75r0 is "1761625.0".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "548095.8974585026".
the Y of the a76r0 is "1761654.2159204248".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "548141.0".
the Y of the a77r0 is "1761588.0".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "548133.8987459367".
the Y of the a78r0 is "1761627.8674571011".

[create a93r0]
a93r0 is a area. "An area (93) in r0".
the printed name of the a93r0 is "Room 0".
Understand "Area 93 in Room 0" as a93r0.
the parent of the a93r0 is r0.
the X of the a93r0 is "549926.0".
the Y of the a93r0 is "1762123.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "549007.0".
the Y of the a28r1 is "1760168.5778301887".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "549373.7290448344".
the Y of the a29r1 is "1760045.0627030537".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "548625.3945333334".
the Y of the a30r1 is "1760040.2686666667".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "548580.0".
the Y of the a81r1 is "1759991.0".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "548611.2137161086".
the Y of the a82r1 is "1760016.633971292".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "548571.0".
the Y of the a85r1 is "1760080.0".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "548601.8730024244".
the Y of the a86r1 is "1760060.6444609193".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "549422.0".
the Y of the a89r1 is "1759991.0".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "549390.7862838916".
the Y of the a90r1 is "1760016.633971292".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "549431.0".
the Y of the a91r1 is "1760080.0".

[create a92r1]
a92r1 is a area. "An area (92) in r1".
the printed name of the a92r1 is "Room 1".
Understand "Area 92 in Room 1" as a92r1.
the parent of the a92r1 is r1.
the X of the a92r1 is "549387.9383292025".
the Y of the a92r1 is "1760061.8770228897".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "546916.8224172432".
the Y of the a19r2 is "1760357.5305011154".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "548275.3853736456".
the Y of the a20r2 is "1760447.6808904759".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "547944.0218346366".
the Y of the a32r2 is "1760084.5039537789".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "547997.0".
the Y of the a65r2 is "1759963.0".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "547961.4166666667".
the Y of the a66r2 is "1760045.8310397554".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "546750.0".
the Y of the a67r2 is "1760273.0".

[create a68r2]
a68r2 is a area. "An area (68) in r2".
the printed name of the a68r2 is "Room 2".
Understand "Area 68 in Room 2" as a68r2.
the parent of the a68r2 is r2.
the X of the a68r2 is "546820.9942396785".
the Y of the a68r2 is "1760311.6157052463".

[create a69r2]
a69r2 is a area. "An area (69) in r2".
the printed name of the a69r2 is "Room 2".
Understand "Area 69 in Room 2" as a69r2.
the parent of the a69r2 is r2.
the X of the a69r2 is "548397.0".
the Y of the a69r2 is "1760399.0".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "548300.4006425082".
the Y of the a70r2 is "1760437.4374839372".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "546269.707266629".
the Y of the a2r3 is "1762061.4777749956".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "546518.8358490563".
the Y of the a3r3 is "1762037.075345912".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "546884.5666747622".
the Y of the a4r3 is "1761940.8403170807".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "546937.3133587773".
the Y of the a5r3 is "1761988.791879405".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "546615.0165977783".
the Y of the a8r3 is "1761973.395123817".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "546881.205995742".
the Y of the a12r3 is "1761862.9993096564".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "546604.0887832437".
the Y of the a13r3 is "1761932.3796623948".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547242.8606168446".
the Y of the a17r3 is "1762059.8602906289".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "546598.3889675064".
the Y of the a22r3 is "1761896.1815944833".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "547291.0".
the Y of the a37r3 is "1762114.0".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547259.9123931625".
the Y of the a38r3 is "1762088.1662393163".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "546982.1111111111".
the Y of the a39r3 is "1762104.0".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "546949.1923615087".
the Y of the a40r3 is "1762105.40408955".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "546480.0".
the Y of the a43r3 is "1762114.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "546511.826460481".
the Y of the a44r3 is "1762105.767353952".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "547300.0".
the Y of the a51r3 is "1762024.0".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "547257.94".
the Y of the a52r3 is "1762042.519".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "546213.0".
the Y of the a57r3 is "1762024.0".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "546251.7029286582".
the Y of the a58r3 is "1762042.477452371".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546223.0".
the Y of the a61r3 is "1762114.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "546253.8949627934".
the Y of the a62r3 is "1762087.7894485788".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "545978.1730804583".
the Y of the a21r4 is "1760859.7005024818".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "545983.0".
the Y of the a26r4 is "1760168.5778301887".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "546349.7290448344".
the Y of the a27r4 is "1760045.0627030537".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "545601.3945333334".
the Y of the a31r4 is "1760040.2686666667".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "546407.0".
the Y of the a63r4 is "1760080.0".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "546363.9383292026".
the Y of the a64r4 is "1760061.87702289".

[create a79r4]
a79r4 is a area. "An area (79) in r4".
the printed name of the a79r4 is "Room 4".
Understand "Area 79 in Room 4" as a79r4.
the parent of the a79r4 is r4.
the X of the a79r4 is "545556.0".
the Y of the a79r4 is "1759991.0".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "545587.2137161086".
the Y of the a80r4 is "1760016.633971292".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "545547.0".
the Y of the a83r4 is "1760080.0".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "545577.8730024244".
the Y of the a84r4 is "1760060.6444609193".

[create a87r4]
a87r4 is a area. "An area (87) in r4".
the printed name of the a87r4 is "Room 4".
Understand "Area 87 in Room 4" as a87r4.
the parent of the a87r4 is r4.
the X of the a87r4 is "546398.0".
the Y of the a87r4 is "1759991.0".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "546366.7862838916".
the Y of the a88r4 is "1760016.6339712923".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "545409.1530110595".
the Y of the a14r5 is "1761937.6714275307".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "545791.915169287".
the Y of the a15r5 is "1762061.6108476832".

[create a18r5]
a18r5 is a area. "An area (18) in r5".
the printed name of the a18r5 is "Room 5".
Understand "Area 18 in Room 5" as a18r5.
the parent of the a18r5 is r5.
the X of the a18r5 is "545044.0175949462".
the Y of the a18r5 is "1762061.5825998154".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "544997.0".
the Y of the a33r5 is "1762114.0".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "545027.915378007".
the Y of the a34r5 is "1762088.1834621995".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "545839.0".
the Y of the a35r5 is "1762114.0".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "545807.9123931623".
the Y of the a36r5 is "1762088.1662393163".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "544988.0".
the Y of the a53r5 is "1762024.0".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "545026.2466592428".
the Y of the a54r5 is "1762042.9003340756".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "545848.0".
the Y of the a55r5 is "1762024.0".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "545808.9448424069".
the Y of the a56r5 is "1762042.8194842408".

[create d0]
d0 is a door. "A door between a16r0 and a28r1".
d0 is south of a16r0 and north of a28r1.

[create d2]
d2 is a door. "A door between a19r2 and a12r3".
d2 is north of a19r2 and south of a12r3.

[create d4]
d4 is a door. "A door between a0r0 and a20r2".
d4 is south of a0r0 and north of a20r2.

[create d6]
d6 is a door. "A door between a22r3 and a21r4".
d6 is southwest of a22r3 and northeast of a21r4.

[create d8]
d8 is a door. "A door between a21r4 and a14r5".
d8 is northwest of a21r4 and southeast of a14r5.

east of a0r0 is west of a1r0.

east of a2r3 is west of a3r3.

northeast of a4r3 is southwest of a5r3.

east of a6r0 is west of a7r0.

southeast of a3r3 is northwest of a8r3.

southeast of a7r0 is northwest of a9r0.

northeast of a10r0 is southwest of a11r0.

south of a4r3 is north of a12r3.

south of a8r3 is north of a13r3.

east of a14r5 is west of a15r5.

east of a1r0 is west of a16r0.

west of a4r3 is east of a8r3.

west of a10r0 is east of a16r0.

east of a5r3 is west of a17r3.

west of a14r5 is east of a18r5.

south of a10r0 is north of a23r0.

north of a24r0 is south of a25r0.

southwest of a9r0 is northeast of a25r0.

south of a13r3 is north of a22r3.

east of a26r4 is west of a27r4.

east of a28r1 is west of a29r1.

west of a28r1 is east of a30r1.

west of a26r4 is east of a31r4.

east of a19r2 is west of a32r2.

southwest of a20r2 is northeast of a32r2.

south of a21r4 is north of a26r4.

southeast of a33r5 is northwest of a34r5.

southwest of a35r5 is northeast of a36r5.

southwest of a37r3 is northeast of a38r3.

west of a39r3 is east of a40r3.

west of a41r0 is east of a42r0.

east of a43r3 is west of a44r3.

east of a45r0 is west of a46r0.

southwest of a47r0 is northeast of a48r0.

northeast of a49r0 is southwest of a50r0.

northwest of a51r3 is southeast of a52r3.

northeast of a53r5 is southwest of a54r5.

northwest of a55r5 is southeast of a56r5.

northeast of a57r3 is southwest of a58r3.

southeast of a59r0 is northwest of a60r0.

southeast of a61r3 is northwest of a62r3.

southwest of a63r4 is northeast of a64r4.

northwest of a65r2 is southeast of a66r2.

northeast of a67r2 is southwest of a68r2.

west of a69r2 is east of a70r2.

north of a71r0 is south of a72r0.

northwest of a73r0 is southeast of a74r0.

northeast of a75r0 is southwest of a76r0.

north of a77r0 is south of a78r0.

northeast of a79r4 is southwest of a80r4.

northeast of a81r1 is southwest of a82r1.

southeast of a83r4 is northwest of a84r4.

southeast of a85r1 is northwest of a86r1.

northwest of a87r4 is southeast of a88r4.

northwest of a89r1 is southeast of a90r1.

southwest of a91r1 is northeast of a92r1.

northeast of a11r0 is southwest of a93r0.

north of a3r3 is south of a44r3.

north of a7r0 is south of a46r0.

southwest of a2r3 is northeast of a58r3.

northwest of a6r0 is southeast of a60r0.

northwest of a2r3 is southeast of a62r3.

northwest of a18r5 is southeast of a34r5.

northeast of a15r5 is southwest of a36r5.

northeast of a17r3 is southwest of a38r3.

north of a5r3 is south of a40r3.

north of a1r0 is south of a42r0.

west of a24r0 is east of a76r0.

southeast of a24r0 is northwest of a78r0.

east of a25r0 is west of a48r0.

west of a0r0 is east of a9r0.

southwest of a6r0 is northeast of a50r0.

southeast of a17r3 is northwest of a52r3.

southwest of a18r5 is northeast of a54r5.

southeast of a15r5 is northwest of a56r5.

southeast of a29r1 is northwest of a90r1.

northeast of a29r1 is southwest of a92r1.

northeast of a27r4 is southwest of a64r4.

southeast of a32r2 is northwest of a66r2.

southwest of a19r2 is northeast of a68r2.

east of a20r2 is west of a70r2.

southwest of a23r0 is northeast of a72r0.

east of a23r0 is west of a74r0.

southwest of a31r4 is northeast of a80r4.

southwest of a30r1 is northeast of a82r1.

northwest of a31r4 is southeast of a84r4.

northwest of a30r1 is southeast of a86r1.

southeast of a27r4 is northwest of a88r4.

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
the player is in a36r5.

the orientation of the player is 6.
