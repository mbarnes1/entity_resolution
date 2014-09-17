#/usr/bin/python
__author__ = 'mbarnes1'


class BlockingScheme(object):
    def __init__(self, records, max_block_size=float('inf')):
        self._max_block_size = max_block_size
        self.strong_blocks = dict()
        self.weak_blocks = dict()
        self._generate_blocks(records)
        self._clean_blocks()
        self._complete_blocks(len(records))

    def _generate_blocks(self, records):
        self._generate_strong_blocks(records)
        self._generate_weak_blocks(records)

    # Inserts ads that were not blocked or whose block was thrown away
    def _complete_blocks(self, number_records):
        used_ads = set()
        for _, ads in self.strong_blocks.iteritems():
            used_ads.update(ads)
        for _, ads in self.weak_blocks.iteritems():
            used_ads.update(ads)
        missing_ads = set(range(0, number_records))
        missing_ads.difference_update(used_ads)
        for ad in missing_ads:
            block = 'singular_ad_' + str(ad)
            self.weak_blocks[block] = {ad}

    def _clean_blocks(self):
        # Remove very large blocks
        toremove = list()
        for block, ads in self.strong_blocks.iteritems():
            blocksize = len(ads)
            if blocksize > self._max_block_size:
                #print 'Throwing away block ', block
                toremove.append(block)
        for remove in toremove:
            del self.strong_blocks[remove]

        toremove = list()
        for block, ads in self.weak_blocks.iteritems():
            blocksize = len(ads)
            if blocksize > self._max_block_size:
                #print 'Throwing away block ', block
                toremove.append(block)
        for remove in toremove:
            del self.weak_blocks[remove]

    def number_of_blocks(self):
        return len(self.weak_blocks) + len(self.strong_blocks)

    def _generate_strong_blocks(self, records):
        for ad, record in records.iteritems():  # loop through all the records
            print 'Strong blocking ad ', ad
            for phone in record.phones:
                b = 'phone_' + phone
                if b in self.strong_blocks:
                    self.strong_blocks[b].add(ad)
                else:
                    self.strong_blocks[b] = {ad}
            for poster in record.posters:
                b = 'poster_' + poster
                if b in self.strong_blocks:
                    self.strong_blocks[b].add(ad)
                else:
                    self.strong_blocks[b] = {ad}
            for email in record.emails:
                b = 'email_' + email
                if b in self.strong_blocks:
                    self.strong_blocks[b].add(ad)
                else:
                    self.strong_blocks[b] = {ad}

    def _generate_weak_blocks(self, records):
        for ad, record in records.iteritems():  # loop through all the records
            print 'Weak blocking ad ', ad
            for site in record.sites:
                b = 'site_' + site
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for state in record.states:
                b = 'state_' + state
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for city in record.cities:
                b = 'city_' + city
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for name in record.names:
                b = 'name_' + name
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for age in record.ages:
                # Same age block
                b = 'age_' + str(age)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                # Block including people 1 year older
                b = 'age+1_' + str(age)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                b = 'age+1_' + str(age-1)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for cost in record.costs:
                b = 'cost_' + cost
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for height in record.heights:  # same height
                b = 'height_' + str(height)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                # Block including people 1 in taller
                b = 'height+1_' + str(height)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                b = 'height+1_' + str(height-1)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for weight in record.weights:  # same height. Get similar heights with all combinations between adjacent blocks
                # Same weight block
                b = 'weight_' + str(weight)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                # Block including people 1 lb heavier
                b = 'weight+1_' + str(weight)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                b = 'weight+1_' + str(weight-1)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                # Block including people 2 lb heavier
                b = 'weight+2_' + str(weight)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
                b = 'weight+2_' + str(weight-2)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for cup in record.cups:
                b = 'cup_' + cup
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for chest in record.chests:
                b = 'chest_' + str(chest)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for waist in record.waists:
                b = 'waist_' + str(waist)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}
            for hip in record.hips:
                b = 'hip_' + str(hip)
                if b in self.weak_blocks:
                    self.weak_blocks[b].add(ad)
                else:
                    self.weak_blocks[b] = {ad}