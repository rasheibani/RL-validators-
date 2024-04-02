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

[create r7]
r7 is a indoor_room. "Room 7".

[create r8]
r8 is a indoor_room. "Room 8".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "545223.1371343823".
the Y of the a6r0 is "2495498.0705300597".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "545298.75".
the Y of the a7r0 is "2495592.8165930705".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545075.2499999998".
the Y of the a32r0 is "2495609.8327083336".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "545038.0".
the Y of the a43r0 is "2495640.0".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "545336.0".
the Y of the a49r0 is "2495640.0".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "545738.6462401489".
the Y of the a16r1 is "2493553.4981471454".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545800.9412801263".
the Y of the a19r1 is "2493554.610558574".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "545658.125".
the Y of the a26r1 is "2493449.5247135414".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "545880.875".
the Y of the a29r1 is "2493450.099474627".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "545621.0".
the Y of the a38r1 is "2493400.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "545918.0".
the Y of the a41r1 is "2493400.0".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "546316.1000000001".
the Y of the a2r2 is "2495589.5239782464".

[create a3r2]
a3r2 is a area. "An area (3) in r2".
the printed name of the a3r2 is "Room 2".
Understand "Area 3 in Room 2" as a3r2.
the parent of the a3r2 is r2.
the X of the a3r2 is "546387.5565103934".
the Y of the a3r2 is "2495502.5378856105".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "546523.8305982074".
the Y of the a8r2 is "2495498.8947038352".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "546596.9".
the Y of the a21r2 is "2495589.466754617".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "546632.0".
the Y of the a45r2 is "2495640.0".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "546281.0".
the Y of the a47r2 is "2495640.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547145.9951063204".
the Y of the a17r3 is "2493554.3375987834".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "547017.0".
the Y of the a22r3 is "2493449.422623434".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "547097.2576810806".
the Y of the a23r3 is "2493553.704645209".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "547227.0".
the Y of the a31r3 is "2493449.3545311806".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547262.0".
the Y of the a35r3 is "2493400.0".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "546982.0".
the Y of the a36r3 is "2493400.0".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "547687.5368908948".
the Y of the a11r4 is "2495498.361141558".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "547834.9999999997".
the Y of the a12r4 is "2495615.693214142".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "547686.4939327079".
the Y of the a14r4 is "2495181.408731028".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "547540.0000000005".
the Y of the a33r4 is "2495613.3077936065".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "547573.0".
the Y of the a44r4 is "2495640.0".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "547805.0".
the Y of the a51r4 is "2495640.0".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "548207.7241818234".
the Y of the a15r5 is "2493553.499538961".

[create a20r5]
a20r5 is a area. "An area (20) in r5".
the printed name of the a20r5 is "Room 5".
Understand "Area 20 in Room 5" as a20r5.
the parent of the a20r5 is r5.
the X of the a20r5 is "548269.9412801259".
the Y of the a20r5 is "2493554.610558574".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "548127.125".
the Y of the a27r5 is "2493449.4190799687".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "548349.875".
the Y of the a28r5 is "2493450.099474627".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "548090.0".
the Y of the a39r5 is "2493400.0".

[create a40r5]
a40r5 is a area. "An area (40) in r5".
the printed name of the a40r5 is "Room 5".
Understand "Area 40 in Room 5" as a40r5.
the parent of the a40r5 is r5.
the X of the a40r5 is "548387.0".
the Y of the a40r5 is "2493400.0".

[create a0r6]
a0r6 is a area. "An area (0) in r6".
the printed name of the a0r6 is "Room 6".
Understand "Area 0 in Room 6" as a0r6.
the parent of the a0r6 is r6.
the X of the a0r6 is "548992.8305982077".
the Y of the a0r6 is "2495498.8947038352".

[create a1r6]
a1r6 is a area. "An area (1) in r6".
the printed name of the a1r6 is "Room 6".
Understand "Area 1 in Room 6" as a1r6.
the parent of the a1r6 is r6.
the X of the a1r6 is "549065.9".
the Y of the a1r6 is "2495589.466754617".

[create a4r6]
a4r6 is a area. "An area (4) in r6".
the printed name of the a4r6 is "Room 6".
Understand "Area 4 in Room 6" as a4r6.
the parent of the a4r6 is r6.
the X of the a4r6 is "548785.1".
the Y of the a4r6 is "2495589.523978247".

[create a5r6]
a5r6 is a area. "An area (5) in r6".
the printed name of the a5r6 is "Room 6".
Understand "Area 5 in Room 6" as a5r6.
the parent of the a5r6 is r6.
the X of the a5r6 is "548856.5565103934".
the Y of the a5r6 is "2495502.5378856105".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "549101.0".
the Y of the a46r6 is "2495640.0".

[create a48r6]
a48r6 is a area. "An area (48) in r6".
the printed name of the a48r6 is "Room 6".
Understand "Area 48 in Room 6" as a48r6.
the parent of the a48r6 is r6.
the X of the a48r6 is "548750.0".
the Y of the a48r6 is "2495640.0".

[create a18r7]
a18r7 is a area. "An area (18) in r7".
the printed name of the a18r7 is "Room 7".
Understand "Area 18 in Room 7" as a18r7.
the parent of the a18r7 is r7.
the X of the a18r7 is "549615.075132821".
the Y of the a18r7 is "2493554.3386380887".

[create a24r7]
a24r7 is a area. "An area (24) in r7".
the printed name of the a24r7 is "Room 7".
Understand "Area 24 in Room 7" as a24r7.
the parent of the a24r7 is r7.
the X of the a24r7 is "549486.0".
the Y of the a24r7 is "2493449.422623434".

[create a25r7]
a25r7 is a area. "An area (25) in r7".
the printed name of the a25r7 is "Room 7".
Understand "Area 25 in Room 7" as a25r7.
the parent of the a25r7 is r7.
the X of the a25r7 is "549566.2576810806".
the Y of the a25r7 is "2493553.704645209".

[create a30r7]
a30r7 is a area. "An area (30) in r7".
the printed name of the a30r7 is "Room 7".
Understand "Area 30 in Room 7" as a30r7.
the parent of the a30r7 is r7.
the X of the a30r7 is "549696.0".
the Y of the a30r7 is "2493449.4606324406".

[create a34r7]
a34r7 is a area. "An area (34) in r7".
the printed name of the a34r7 is "Room 7".
Understand "Area 34 in Room 7" as a34r7.
the parent of the a34r7 is r7.
the X of the a34r7 is "549731.0".
the Y of the a34r7 is "2493400.0".

[create a37r7]
a37r7 is a area. "An area (37) in r7".
the printed name of the a37r7 is "Room 7".
Understand "Area 37 in Room 7" as a37r7.
the parent of the a37r7 is r7.
the X of the a37r7 is "549451.0".
the Y of the a37r7 is "2493400.0".

[create a9r8]
a9r8 is a area. "An area (9) in r8".
the printed name of the a9r8 is "Room 8".
Understand "Area 9 in Room 8" as a9r8.
the parent of the a9r8 is r8.
the X of the a9r8 is "550078.4999999993".
the Y of the a9r8 is "2495591.129062261".

[create a10r8]
a10r8 is a area. "An area (10) in r8".
the printed name of the a10r8 is "Room 8".
Understand "Area 10 in Room 8" as a10r8.
the parent of the a10r8 is r8.
the X of the a10r8 is "550152.0274643078".
the Y of the a10r8 is "2495502.3687288044".

[create a13r8]
a13r8 is a area. "An area (13) in r8".
the printed name of the a13r8 is "Room 8".
Understand "Area 13 in Room 8" as a13r8.
the parent of the a13r8 is r8.
the X of the a13r8 is "550297.4999999998".
the Y of the a13r8 is "2495609.7869122024".

[create a42r8]
a42r8 is a area. "An area (42) in r8".
the printed name of the a42r8 is "Room 8".
Understand "Area 42 in Room 8" as a42r8.
the parent of the a42r8 is r8.
the X of the a42r8 is "550334.0".
the Y of the a42r8 is "2495640.0".

[create a50r8]
a50r8 is a area. "An area (50) in r8".
the printed name of the a50r8 is "Room 8".
Understand "Area 50 in Room 8" as a50r8.
the parent of the a50r8 is r8.
the X of the a50r8 is "550042.0".
the Y of the a50r8 is "2495640.0".

[create d0]
d0 is a door. "A door between a8r2 and a23r3".
d0 is south of a8r2 and north of a23r3.

[create d2]
d2 is a door. "A door between a0r6 and a25r7".
d2 is south of a0r6 and north of a25r7.

[create d4]
d4 is a door. "A door between a14r4 and a15r5".
d4 is south of a14r4 and north of a15r5.

[create d6]
d6 is a door. "A door between a6r0 and a16r1".
d6 is south of a6r0 and north of a16r1.

[create d8]
d8 is a door. "A door between a17r3 and a14r4".
d8 is northeast of a17r3 and southwest of a14r4.

[create d10]
d10 is a door. "A door between a18r7 and a10r8".
d10 is north of a18r7 and south of a10r8.

[create d12]
d12 is a door. "A door between a19r1 and a3r2".
d12 is north of a19r1 and south of a3r2.

[create d14]
d14 is a door. "A door between a20r5 and a5r6".
d14 is north of a20r5 and south of a5r6.

northeast of a0r6 is southwest of a1r6.

southeast of a2r2 is northwest of a3r2.

southeast of a4r6 is northwest of a5r6.

northeast of a6r0 is southwest of a7r0.

east of a3r2 is west of a8r2.

west of a0r6 is east of a5r6.

southeast of a9r8 is northwest of a10r8.

northeast of a11r4 is southwest of a12r4.

northeast of a10r8 is southwest of a13r8.

south of a11r4 is north of a14r4.

northeast of a8r2 is southwest of a21r2.

northeast of a22r3 is southwest of a23r3.

northeast of a24r7 is southwest of a25r7.

southwest of a16r1 is northeast of a26r1.

southwest of a15r5 is northeast of a27r5.

southeast of a20r5 is northwest of a28r5.

southeast of a19r1 is northwest of a29r1.

southeast of a18r7 is northwest of a30r7.

southeast of a17r3 is northwest of a31r3.

northwest of a6r0 is southeast of a32r0.

northwest of a11r4 is southeast of a33r4.

southeast of a30r7 is northwest of a34r7.

southeast of a31r3 is northwest of a35r3.

southwest of a22r3 is northeast of a36r3.

southwest of a24r7 is northeast of a37r7.

southwest of a26r1 is northeast of a38r1.

southwest of a27r5 is northeast of a39r5.

southeast of a28r5 is northwest of a40r5.

southeast of a29r1 is northwest of a41r1.

northeast of a13r8 is southwest of a42r8.

northwest of a32r0 is southeast of a43r0.

northeast of a33r4 is southwest of a44r4.

northeast of a21r2 is southwest of a45r2.

northeast of a1r6 is southwest of a46r6.

northwest of a2r2 is southeast of a47r2.

northwest of a4r6 is southeast of a48r6.

northeast of a7r0 is southwest of a49r0.

northwest of a9r8 is southeast of a50r8.

northwest of a12r4 is southeast of a51r4.

west of a17r3 is east of a23r3.

west of a18r7 is east of a25r7.

east of a16r1 is west of a19r1.

east of a15r5 is west of a20r5.

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
the player is in a41r1.

the orientation of the player is 6.
