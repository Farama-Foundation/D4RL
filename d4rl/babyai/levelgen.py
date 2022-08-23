import random
from collections import OrderedDict
import gym
from gym import spaces
from d4rl.gym_minigrid.roomgrid import RoomGrid
from .verifier import *
import string


class String(gym.Space):
    def __init__(
            self,
            length=None,
            min_length=1,
            max_length=180,
    ):
        super(String, self).__init__()
        self.length = length
        self.min_length = min_length
        self.max_length = max_length
        self.letters = string.ascii_letters + " .,!-"

    def sample(self, **kwargs):
        length = random.randint(self.min_length, self.max_length)
        string = ""
        for i in range(length):
            letter = random.choice(self.letters)
            string += letter
        return string

    def contains(self, x):
        return type(x) is str and len(x) > self.min_length and len(x) < self.max_length


class RejectSampling(Exception):
    """
    Exception used for rejection sampling
    """

    pass


class RoomGridLevel(RoomGrid):
    """
    Base for levels based on RoomGrid
    A level, given a random seed, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """
    metadata={"render_modes": []}

    def __init__(
            self,
            room_size=8,
            **kwargs
    ):
        super().__init__(
            room_size=room_size,
            **kwargs
        )
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': image_space,
            'direction': spaces.Discrete(4),
            'mission': String(),
        })

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size ** 2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        num_navs = self.num_navs_needed(self.instrs)
        self.max_steps = num_navs * nav_time_maze

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # If we drop an object, we need to update its position in the environment
        if action == self.actions.drop:
            self.update_objs_poss()

        # If we've successfully completed the mission
        status = self.instrs.verify(action)

        if status == 'success':
            done = True
            reward = self._reward()
        elif status == 'failure':
            done = True
            reward = 0

        return obs, reward, done, info

    def update_objs_poss(self, instr=None):
        if instr is None:
            instr = self.instrs
        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr) or isinstance(instr, AfterInstr):
            self.update_objs_poss(instr.instr_a)
            self.update_objs_poss(instr.instr_b)
        else:
            instr.update_objs_poss()

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super()._gen_grid(width, height)

                # Generate the mission
                self.gen_mission()

                # Validate the instructions
                self.validate_instrs(self.instrs)

            except RecursionError as error:
                print('Timeout during mission generation:', error)
                continue

            except RejectSampling as error:
                # print('Sampling rejected:', error)
                continue

            break

        # Generate the surface form for the instructions
        self.surface = self.instrs.surface(self)
        self.mission = self.surface

    def validate_instrs(self, instr):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        if hasattr(self, 'unblocking') and self.unblocking:
            colors_of_locked_doors = []
            for i in range(self.num_cols):
                for j in range(self.num_rows):
                    room = self.get_room(i, j)
                    for door in room.doors:
                        if door and door.is_locked:
                            colors_of_locked_doors.append(door.color)

        if isinstance(instr, PutNextInstr):
            # Resolve the objects referenced by the instruction
            instr.reset_verifier(self)

            # Check that the objects are not already next to each other
            if set(instr.desc_move.obj_set).intersection(
                    set(instr.desc_fixed.obj_set)):
                raise RejectSampling(
                    "there are objects that match both lhs and rhs of PutNext")
            if instr.objs_next():
                raise RejectSampling('objs already next to each other')

            # Check that we are not asking to move an object next to itself
            move = instr.desc_move
            fixed = instr.desc_fixed
            if len(move.obj_set) == 1 and len(fixed.obj_set) == 1:
                if move.obj_set[0] is fixed.obj_set[0]:
                    raise RejectSampling('cannot move an object next to itself')

        if isinstance(instr, ActionInstr):
            if not hasattr(self, 'unblocking') or not self.unblocking:
                return
            # TODO: either relax this a bit or make the bot handle this super corner-y scenarios
            # Check that the instruction doesn't involve a key that matches the color of a locked door
            potential_objects = ('desc', 'desc_move', 'desc_fixed')
            for attr in potential_objects:
                if hasattr(instr, attr):
                    obj = getattr(instr, attr)
                    if obj.type == 'key' and obj.color in colors_of_locked_doors:
                        raise RejectSampling('cannot do anything with/to a key that can be used to open a door')
            return

        if isinstance(instr, SeqInstr):
            self.validate_instrs(instr.instr_a)
            self.validate_instrs(instr.instr_b)
            return

        assert False, "unhandled instruction type"

    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError

    @property
    def level_name(self):
        return self.__class__.level_name

    @property
    def gym_id(self):
        return self.__class__.gym_id

    def num_navs_needed(self, instr):
        """
        Compute the maximum number of navigations needed to perform
        a simple or complex instruction
        """

        if isinstance(instr, PutNextInstr):
            return 2

        if isinstance(instr, ActionInstr):
            return 1

        if isinstance(instr, SeqInstr):
            na = self.num_navs_needed(instr.instr_a)
            nb = self.num_navs_needed(instr.instr_b)
            return na + nb

    def open_all_doors(self):
        """
        Open all the doors in the maze
        """

        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        door.is_open = True

    def check_objs_reachable(self, raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """

        # Reachable positions
        reachable = set()

        # Work list
        stack = [self.agent_pos]

        while len(stack) > 0:
            i, j = stack.pop()

            if i < 0 or i >= self.grid.width or j < 0 or j >= self.grid.height:
                continue

            if (i, j) in reachable:
                continue

            # This position is reachable
            reachable.add((i, j))

            cell = self.grid.get(i, j)

            # If there is something other than a door in this cell, it
            # blocks reachability
            if cell and cell.type != 'door':
                continue

            # Visit the horizontal and vertical neighbors
            stack.append((i + 1, j))
            stack.append((i - 1, j))
            stack.append((i, j + 1))
            stack.append((i, j - 1))

        # Check that all objects are reachable
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)

                if not cell or cell.type == 'wall':
                    continue

                if (i, j) not in reachable:
                    if not raise_exc:
                        return False
                    raise RejectSampling('unreachable object at ' + str((i, j)))

        # All objects reachable
        return True


class LevelGen(RoomGridLevel):
    """
    Level generator which attempts to produce every possible sentence in
    the baby language as an instruction.
    """

    def __init__(
            self,
            room_size=8,
            num_rows=3,
            num_cols=3,
            num_dists=18,
            locked_room_prob=0.5,
            locations=True,
            unblocking=True,
            implicit_unlock=True,
            action_kinds=['goto', 'pickup', 'open', 'putnext'],
            instr_kinds=['action', 'and', 'seq'],
            seed=None
    ):
        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds

        self.locked_room = None

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            seed=seed
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds,
            instr_kinds=self.instr_kinds
        )

    def add_locked_room(self):
        # Until we've successfully added a locked room
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)

            # Don't add a locked door in an external wall
            if self.locked_room.neighbors[door_idx] is None:
                continue

            door, _ = self.add_door(
                i, j,
                door_idx,
                locked=True
            )

            # Done adding locked room
            break

        # Until we find a room to put the key
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is self.locked_room:
                continue

            self.add_object(i, j, 'key', door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES, max_tries=100):
        """
        Generate a random object descriptor
        """

        num_tries = 0

        # Keep trying until we find a matching object
        while True:
            if num_tries > max_tries:
                raise RecursionError('failed to find suitable object')
            num_tries += 1

            color = self._rand_elem([None, *colors])
            type = self._rand_elem(types)

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)

            desc = ObjDesc(type, color, loc)

            # Find all objects matching the descriptor
            objs, poss = desc.find_matching_objs(self)

            # The description must match at least one object
            if len(objs) == 0:
                continue

            # If no implicit unlocking is required
            if not self.implicit_unlock and self.locked_room:
                # Check that at least one object is not in the locked room
                pos_not_locked = list(filter(
                    lambda p: not self.locked_room.pos_inside(*p),
                    poss
                ))

                if len(pos_not_locked) == 0:
                    continue

            # Found a valid object description
            return desc

    def rand_instr(
            self,
            action_kinds,
            instr_kinds,
            depth=0
    ):
        """
        Generate random instructions
        """

        kind = self._rand_elem(instr_kinds)

        if kind == 'action':
            action = self._rand_elem(action_kinds)

            if action == 'goto':
                return GoToInstr(self.rand_obj())
            elif action == 'pickup':
                return PickupInstr(self.rand_obj(types=OBJ_TYPES_NOT_DOOR))
            elif action == 'open':
                return OpenInstr(self.rand_obj(types=['door']))
            elif action == 'putnext':
                return PutNextInstr(
                    self.rand_obj(types=OBJ_TYPES_NOT_DOOR),
                    self.rand_obj()
                )

            assert False

        elif kind == 'and':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth + 1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action'],
                depth=depth + 1
            )
            return AndInstr(instr_a, instr_b)

        elif kind == 'seq':
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth + 1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=['action', 'and'],
                depth=depth + 1
            )

            kind = self._rand_elem(['before', 'after'])

            if kind == 'before':
                return BeforeInstr(instr_a, instr_b)
            elif kind == 'after':
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False


# Dictionary of levels, indexed by name, lexically sorted
level_dict = OrderedDict()


def register_levels(module_name, globals):
    """
    Register OpenAI gym environments for all levels in a file
    """

    # Iterate through global names
    for global_name in sorted(list(globals.keys())):
        if not global_name.startswith('Level_'):
            continue

        level_name = global_name.split('Level_')[-1]
        level_class = globals[global_name]

        # Register the levels with OpenAI Gym
        gym_id = 'BabyAI-%s-v0' % level_name
        entry_point = '%s:%s' % (module_name, global_name)
        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        # Add the level to the dictionary
        level_dict[level_name] = level_class

        # Store the name and gym id on the level class
        level_class.level_name = level_name
        level_class.gym_id = gym_id


def test():
    for idx, level_name in enumerate(level_dict.keys()):
        print('Level %s (%d/%d)' % (level_name, idx + 1, len(level_dict)))

        level = level_dict[level_name]

        # Run the mission for a few episodes
        rng = random.Random(0)
        num_episodes = 0
        for i in range(0, 15):
            mission = level(seed=i)

            # Check that the surface form was generated
            assert isinstance(mission.surface, str)
            assert len(mission.surface) > 0
            obs = mission.reset()
            assert obs['mission'] == mission.surface

            # Reduce max_steps because otherwise tests take too long
            mission.max_steps = min(mission.max_steps, 200)

            # Check for some known invalid patterns in the surface form
            import re
            surface = mission.surface
            assert not re.match(r".*pick up the [^ ]*door.*", surface), surface

            while True:
                action = rng.randint(0, mission.action_space.n - 1)
                obs, reward, done, info = mission.step(action)
                if done:
                    obs = mission.reset()
                    break

            num_episodes += 1

        # The same seed should always yield the same mission
        m0 = level(seed=0)
        m1 = level(seed=0)
        grid1 = m0.unwrapped.grid
        grid2 = m1.unwrapped.grid
        assert grid1 == grid2
        assert m0.surface == m1.surface

    # Check that gym environment names were registered correctly
    gym.make('BabyAI-1RoomS8-v0')
    gym.make('BabyAI-BossLevel-v0')
