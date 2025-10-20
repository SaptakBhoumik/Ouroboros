#include "tensor.hpp"
namespace Ouroboros{
NDRange::NDRange(Shape s,std::set<uint64_t> axis) : shape(s), axis(axis) {}

void NDRange::Iterator1::reset(std::uint64_t off) {
    offset = off;
    end_flag = false;
    for (std::uint64_t i = 0; i < index.size(); ++i) {
        index[i] = 0;
    }
}
NDRange::Iterator1::Iterator1(const std::vector<uint64_t>* shape_, const std::vector<uint64_t>* weight_, bool end,std::uint64_t off)
            : shape(shape_), weight(weight_), end_flag(end), offset(off){
            index.assign(shape->size(), 0);
            if (end_flag && !shape->empty())
                index[0] = (*shape)[0]; // make it “past-the-end”
}

NDRange::Iterator1& NDRange::Iterator1::operator++() {
    for (std::int64_t i = index.size() - 1; i >= 0; --i) {
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
bool NDRange::Iterator1::operator!=(const NDRange::Iterator1& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return index != other.index;
}


void NDRange::Range::reset(std::uint64_t off){
    it_start.reset(off);
    it_end.reset(off);
}
NDRange::Range::Range(const std::vector<uint64_t>* shape_, const std::vector<uint64_t>* weight_,std::uint64_t off)
            : it_start(shape_, weight_, false, off), it_end(shape_, weight_, true, off) {}

NDRange::Iterator1 NDRange::Range::begin() const{
    return it_start;
}
NDRange::Iterator1 NDRange::Range::end() const{
    return it_end;
}

NDRange::Iterator0::Iterator0(const Shape* shape_,const std::set<std::uint64_t>& axis, bool end)
            : end_flag(end){
    for (std::uint64_t i = 0; i < shape_->dim(); i++) {
        if (axis.find(i) != axis.end()) {
            strides0.push_back(shape_->get_stride(i));
            shape0.push_back(shape_->operator[](i));
        } else {
            strides1.push_back(shape_->get_stride(i));
            shape1.push_back(shape_->operator[](i));
        }
    }
    it0 = Iterator1(&shape0, &strides0, false);
    it1 = Range(&shape1, &strides1);
}

NDRange::Iterator0& NDRange::Iterator0::operator++() {
    ++it0;
    it1.reset(*it0);
    return *this;
}
bool NDRange::Iterator0::operator!=(const NDRange::Iterator0& other) const {
    if (end_flag != other.end_flag) return true;
    if (end_flag) return false;
    return it0 != other.it0;
}

NDRange::Iterator0 NDRange::begin() const { return Iterator0(&shape, axis, false); }
NDRange::Iterator0 NDRange::end()   const { return Iterator0(&shape, axis, true);  }


IdxIterator::IdxIterator(Shape s,std::unordered_map<std::uint64_t,std::uint64_t> fixed_indices) {
    is_fixed = std::vector<int64_t>(s.dim(), -1);
    for (const auto& [idx, val] : fixed_indices) {
        is_fixed[idx] = static_cast<int64_t>(val);
    }
    for (std::uint64_t i = 0; i < s.dim(); ++i) {
        shape.push_back(s[i]);
        weight.push_back(s.get_stride(i));
    }
}
IdxIterator::Iterator::Iterator(const std::vector<uint64_t>* shape_, const std::vector<int64_t>* is_fixed_, const std::vector<uint64_t>* weight_,bool end): 
    shape(shape_), is_fixed(is_fixed_), weight(weight_), end_flag(end){
    index.assign(shape->size(), 0);
    for (std::uint64_t i = 0; i < shape->size(); i++) {
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

IdxIterator2::IdxIterator2(const Shape& shape_,const std::vector<uint64_t>& start_,const std::vector<uint64_t>& end_,const std::vector<uint64_t>& step_):
    start(start_), _end(end_), step(step_){
    for (std::uint64_t i = 0; i < shape_.dim(); i++) {
        weight.push_back(shape_.get_stride(i));
    }
}
IdxIterator2::Iterator::Iterator(const std::vector<uint64_t>* start_,const std::vector<uint64_t>* end_, const std::vector<uint64_t>* step_,
        const std::vector<uint64_t>* weight_,bool end): start(start_), end(end_), step(step_), weight(weight_), end_flag(end){
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