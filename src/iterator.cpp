#include "tensor.hpp"
namespace Ouroboros{
NDRange::NDRange(Shape s,std::unordered_set<size_t> axis) : shape(s), axis(axis) {}

void NDRange::Iterator1::reset(std::size_t off, bool end) {
    offset = off;
    end_flag = end;
    index.assign(shape ? shape->size() : 0, 0);
    if (end_flag && shape && !shape->empty()) {
        // mark "past-the-end" lexicographically: first dim = size
        index[0] = (*shape)[0];
    }
}

NDRange::Iterator1::Iterator1(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_, bool end, std::size_t off)
    : shape(shape_), weight(weight_), offset(off), end_flag(end) {
    index.assign(shape ? shape->size() : 0, 0);
    if (end_flag && shape && !shape->empty())
        index[0] = (*shape)[0]; // past-the-end marker
}

NDRange::Iterator1& NDRange::Iterator1::operator++() {
    if (!shape || shape->empty()) {
        end_flag = true;
        return *this;
    }
    for (std::int64_t i = static_cast<std::int64_t>(index.size()) - 1; i >= 0; --i) {
        ++index[i];
        if (index[i] < (*shape)[i]) {
            offset += (*weight)[i];
            return *this;
        }
        // wrapped: remove previous contribution and set to 0
        offset -= ( (*shape)[i] - 1 ) * (*weight)[i];
        index[i] = 0;
    }
    // we've overflowed the most-significant digit -> past-the-end
    end_flag = true;
    // make consistent "past-the-end" index state
    if (!shape->empty()) index[0] = (*shape)[0];
    return *this;
}

bool NDRange::Iterator1::operator!=(const NDRange::Iterator1& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return index != other.index;
}



void NDRange::Range::reset(std::size_t off){
    // preserve the meaning of it_end being past-the-end
    it_start.reset(off, false);
    it_end.reset(off, true);
    // index doesn't change here; index holds the outer fixed-index vector
}

NDRange::Range::Range(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_, std::vector<std::size_t> index_, std::size_t off)
    : it_start(shape_, weight_, false, off),
      it_end(shape_, weight_, true, off),
      index(std::move(index_)),
      offset(off) {}

NDRange::Iterator1 NDRange::Range::begin() const{
    return it_start;
}
NDRange::Iterator1 NDRange::Range::end() const{
    return it_end;
}

NDRange::Iterator0::Iterator0(const Shape* shape_, const std::unordered_set<std::size_t>& axis, bool end)
    : end_flag(end)
{
    for (std::size_t i = 0; i < shape_->dim(); i++) {
        if (axis.find(i) != axis.end()) {
            strides0.push_back(shape_->get_stride(i));
            shape0.push_back(shape_->operator[](i));
        } else {
            strides1.push_back(shape_->get_stride(i));
            shape1.push_back(shape_->operator[](i));
        }
    }
    // Construct inner iterator with same 'end' meaning so comparisons are correct
    it0 = Iterator1(&shape0, &strides0, end, 0);
    // Range must be constructed with the current outer index (may be past-the-end when end==true)
    it1 = Range(&shape1, &strides1, it0.get_index(), *it0);
}

NDRange::Iterator0& NDRange::Iterator0::operator++() {
    ++it0;
    // Reset the inner range using the new offset; Range::reset will set it_end correctly
    it1.reset(*it0);
    return *this;
}

bool NDRange::Iterator0::operator!=(const NDRange::Iterator0& other) const {
    // The true iteration state is the inner iterator it0.
    return it0 != other.it0;
}


NDRange::Iterator0 NDRange::begin() const { return Iterator0(&shape, axis, false); }
NDRange::Iterator0 NDRange::end()   const { return Iterator0(&shape, axis, true);  }


IdxIterator::IdxIterator(Shape s,std::unordered_map<std::size_t,std::size_t> fixed_indices) {
    is_fixed = std::vector<int64_t>(s.dim(), -1);
    for (const auto& [idx, val] : fixed_indices) {
        is_fixed[idx] = static_cast<int64_t>(val);
    }
    for (std::size_t i = 0; i < s.dim(); ++i) {
        shape.push_back(s[i]);
        weight.push_back(s.get_stride(i));
    }
}
IdxIterator::Iterator::Iterator(const std::vector<size_t>* shape_, const std::vector<int64_t>* is_fixed_, const std::vector<size_t>* weight_,bool end): 
    shape(shape_), is_fixed(is_fixed_), weight(weight_), end_flag(end){
    index.assign(shape->size(), 0);
    for (std::size_t i = 0; i < shape->size(); i++) {
        if ((*is_fixed)[i] >= 0) {
            index[i] = (*is_fixed)[i];
            offset += index[i] * (*weight)[i];
        }
    }
}
IdxIterator::Iterator& IdxIterator::Iterator::operator++(){
    for (std::int64_t i = index.size() - 1; i >= 0; --i) {
        if ((*is_fixed)[i] >= 0) {
            continue;
        }
        if (++index[i] < (*shape)[i]){
            offset += (*weight)[i];
            return *this;
        }
        offset -= (index[i]-1) * (*weight)[i];
        index[i] = 0;
    }
    end_flag = true;
    return *this;
}
bool IdxIterator::Iterator::operator!=(const Iterator& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return index != other.index;
}

IdxIterator::Iterator IdxIterator::begin() const{return Iterator(&shape, &is_fixed, &weight, false);}
IdxIterator::Iterator IdxIterator::end()  const{return Iterator(&shape, &is_fixed, &weight, true);}

IdxIterator2::IdxIterator2(const Shape& shape_,const std::vector<size_t>& start_,const std::vector<size_t>& end_,const std::vector<size_t>& step_):
    start(start_), _end(end_), step(step_){
    for (std::size_t i = 0; i < shape_.dim(); i++) {
        weight.push_back(shape_.get_stride(i));
    }
}
IdxIterator2::Iterator::Iterator(const std::vector<size_t>* start_,const std::vector<size_t>* end_, const std::vector<size_t>* step_,
        const std::vector<size_t>* weight_,bool end): start(start_), end(end_), step(step_), weight(weight_), end_flag(end){
    index = *start;
}
IdxIterator2::Iterator& IdxIterator2::Iterator::operator++(){
    for (std::int64_t i = index.size() - 1; i >= 0; --i) {
        if ((index[i] + step->at(i)) < end->at(i)) {
            index[i] += step->at(i);
            offset += step->at(i) * weight->at(i);
            return *this;
        }
        offset -= (index[i] - start->at(i)) * (*weight)[i];
        index[i] = start->at(i);
    }
    end_flag = true;
    return *this;
}
bool IdxIterator2::Iterator::operator!=(const Iterator& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return index != other.index;
}
IdxIterator2::Iterator IdxIterator2::begin() const {return Iterator(&start, &_end, &step, &weight, false);}
IdxIterator2::Iterator IdxIterator2::end()   const{ return Iterator(&start, &_end, &step, &weight, true);}
}