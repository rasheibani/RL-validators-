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

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "500920.0".
the Y of the a4r0 is "1686080.0".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "500963.3333333333".
the Y of the a5r0 is "1686035.0".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "501836.0".
the Y of the a8r0 is "1685820.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "501885.1111111111".
the Y of the a9r0 is "1685863.3333333335".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "504496.0".
the Y of the a10r0 is "1685820.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "504449.8461538461".
the Y of the a11r0 is "1685863.3333333335".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "501050.0".
the Y of the a12r0 is "1685082.256610577".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "502850.0".
the Y of the a14r0 is "1685919.096718371".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "503166.0".
the Y of the a15r0 is "1685922.1662973515".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "503296.0".
the Y of the a0r1 is "1684020.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "503252.6666666666".
the Y of the a1r1 is "1684067.3684210526".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "502380.0".
the Y of the a2r2 is "1684020.0".

[create a3r2]
a3r2 is a area. "An area (3) in r2".
the printed name of the a3r2 is "Room 2".
Understand "Area 3 in Room 2" as a3r2.
the parent of the a3r2 is r2.
the X of the a3r2 is "502336.6666666667".
the Y of the a3r2 is "1684065.0".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "500580.0".
the Y of the a6r2 is "1684020.0".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "500536.6666666667".
the Y of the a7r2 is "1684065.0".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "502250.0".
the Y of the a13r2 is "1685017.743389423".

[create d0]
d0 is a door. "A door between a12r0 and a13r2".
d0 is east of a12r0 and west of a13r2.

[create d1]
d1 is a door. "A door between a14r0 and a13r2".
d1 is southwest of a14r0 and northeast of a13r2.

[create d2]
d2 is a door. "A door between a12r0 and a7r2".
d2 is southwest of a12r0 and northeast of a7r2.

northwest of a0r1 is southeast of a1r1.

northwest of a2r2 is southeast of a3r2.

southeast of a4r0 is northwest of a5r0.

northwest of a6r2 is southeast of a7r2.

northeast of a8r0 is southwest of a9r0.

northwest of a10r0 is southeast of a11r0.

east of a14r0 is west of a15r0.

north of a3r2 is south of a13r2.

south of a5r0 is north of a12r0.

east of a9r0 is west of a14r0.

west of a11r0 is east of a15r0.

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
the player is in a13r2.

the orientation of the player is 6.
